"""
Fine-tune een bi-encoder op (review -> parfum-passage) paren. Vereist GPU.

- BASE_MODEL = bge-base (768-dim)
- hard negatives: per anchor een 'near-miss' passage minen (naast in-batch negatives)
- by-perfume split: eval-parfums volledig buiten de training (geen leakage)
- CachedMultipleNegativesRankingLoss: groot effectief batch op beperkte VRAM
- IR-evaluator (Recall@10 op held-out) per epoch; beste checkpoint wordt bewaard

Gebruik: python finetune_embeddings.py  ->  models/bge-fragrance-v2
"""
from __future__ import annotations

import ast
import html
import random
import re

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ---------------------------------------------------------------- config
BASE_MODEL   = "BAAI/bge-base-en-v1.5"
OUT_DIR      = "./models/bge-fragrance-v2"     # fine-tuned model komt hier
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FILES = [
    "fragrantica_data/perfumes_table.csv",
    "fragrantica_data/fragnantica_missing_filled.csv",
    "fragrantica_data/missing_part1_filled.csv",
    "fragrantica_data/missing_part2_filled.csv",
    "fragrantica_data/missing_part3_filled.csv",
]

MIN_REVIEW_LEN        = 80
MAX_REVIEW_CHARS      = 512
MAX_REVIEWS_KEPT      = 8       # per parfum bewaard (langste-eerst); train pakt er een paar van
MAX_PAIRS_PER_PERFUME = 4       # train-paren per parfum
MAX_SEQ_LEN           = 128

EPOCHS      = 2
BATCH       = 128              # effectief aantal in-batch negatives
MINI_BATCH  = 16              # past-in-VRAM brok; verlaag bij OOM
LR          = 2e-5

EVAL_PERFUMES    = 1000        # volledig held-out parfums (queries + relevante passage)
EVAL_DISTRACTORS = 5000        # extra train-passages in de eval-corpus (realistischer)
NUM_NEG          = 1           # hard negatives per paar
SEED             = 42

RECALL_KEY = "frag_cosine_recall@10"   # metric-naam van de IR-evaluator (name="frag")

_WS = re.compile(r"\s+")


#  helpers
def clean(t) -> str:
    return _WS.sub(" ", html.unescape(str(t))).strip()


def review_list(raw) -> list[str]:
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if s.startswith("["):
        try:
            return [str(r) for r in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return [s]
    return [s] if s else []


def notes_str(raw) -> str:
    if pd.isna(raw) or not str(raw).strip():
        return ""
    s = str(raw).strip()
    if s.startswith("["):
        try:
            return ", ".join(str(n) for n in ast.literal_eval(s))
        except (ValueError, SyntaxError):
            pass
    return s


def load_records(files: list[str]) -> list[tuple[str, str, list[str]]]:
    """-> [(url, passage, [schone reviews, langste-eerst])] voor parfums met notes + reviews.
    passage == de fine-tune 'positive' (zelfde vorm als fragrances_notes)."""
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except FileNotFoundError:
            print("overslaan (niet gevonden):", f)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="url")
    df = df[df["notes"].notna() & df["reviews"].notna()]

    records = []
    for _, row in df.iterrows():
        passage = (f"{row.get('title', '')}. Brand: {row.get('designer', '')}. "
                   f"Notes: {notes_str(row.get('notes'))}.")
        revs = [clean(r)[:MAX_REVIEW_CHARS] for r in review_list(row.get("reviews"))]
        revs = [r for r in revs if len(r) >= MIN_REVIEW_LEN]
        revs.sort(key=len, reverse=True)
        if revs:
            records.append((str(row.get("url")), passage, revs[:MAX_REVIEWS_KEPT]))
    return records


def to_pairs(records) -> Dataset:
    """Train-paren: (anchor = prefix+review, positive = passage). Prefix alleen op de
    query-kant (BGE-conventie); de passage is de document-kant."""
    anchors, positives = [], []
    for _url, passage, revs in records:
        for rv in revs[:MAX_PAIRS_PER_PERFUME]:
            anchors.append(QUERY_PREFIX + rv)
            positives.append(passage)
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def build_ir_evaluator(eval_records, train_records) -> InformationRetrievalEvaluator:
    """Review-als-query op held-out parfums. corpus = held-out passages + een sample
    train-passages als afleiders, zodat Recall@10 realistisch is."""
    queries, relevant, corpus = {}, {}, {}
    for k, (_url, passage, revs) in enumerate(eval_records):
        pid, qid = f"p{k}", f"q{k}"
        corpus[pid] = passage
        queries[qid] = QUERY_PREFIX + revs[0]   # langste held-out review als query
        relevant[qid] = {pid}
    sample = random.Random(SEED).sample(
        train_records, min(EVAL_DISTRACTORS, len(train_records)))
    for k, (_url, passage, _revs) in enumerate(sample):
        corpus[f"d{k}"] = passage
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="frag", show_progress_bar=False,
        accuracy_at_k=[1, 10], precision_recall_at_k=[1, 5, 10],
        mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
    )


def mine(train_ds: Dataset, model: SentenceTransformer) -> Dataset:
    """Voeg per (anchor, positive) een HARD negative toe (near-miss passage).
    Valt terug op de paren zonder expliciete negatives als de util ontbreekt."""
    try:
        from sentence_transformers.util import mine_hard_negatives
    except ImportError:
        print("LET OP: mine_hard_negatives niet beschikbaar - train zonder expliciete "
              "hard negatives (alleen in-batch). Upgrade sentence-transformers voor de winst.")
        return train_ds
    print("hard negatives minen...")
    return mine_hard_negatives(
        train_ds, model,
        anchor_column_name="anchor", positive_column_name="positive",
        num_negatives=NUM_NEG,
        range_min=10,
        range_max=60,
        max_score=0.85,
        sampling_strategy="top",
        batch_size=256,
        output_format="triplet",
        use_faiss=True,          # <-- dit lost je OOM op
        verbose=True,
    )


# ---------------------------------------------------------------- train
def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    records = load_records(FILES)
    random.Random(SEED).shuffle(records)
    eval_records, train_records = records[:EVAL_PERFUMES], records[EVAL_PERFUMES:]
    print(f"{len(records):,} parfums met reviews | train {len(train_records):,} | "
          f"eval (held-out) {len(eval_records):,}")

    train_ds = to_pairs(train_records)
    print(f"{len(train_ds):,} (review -> passage) trainingsparen")

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LEN

    evaluator = build_ir_evaluator(eval_records, train_records)
    base = evaluator(model)
    print(f"\nBASELINE ({BASE_MODEL}) Recall@10: {base.get(RECALL_KEY, float('nan')):.3f}")
    print("   (eval-metric keys:", [k for k in base if 'recall' in k], ")")

    train_ds = mine(train_ds, model)   # met de (nog ongetrainde) base mining = semi-hard
    print(f"train-kolommen: {train_ds.column_names}")

    try:
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH)
        print(f"loss: CachedMNRL (mini_batch={MINI_BATCH}, effectief batch={BATCH})")
    except (AttributeError, TypeError):
        loss = losses.MultipleNegativesRankingLoss(model)
        print("loss: MultipleNegativesRankingLoss (CachedMNRL niet beschikbaar)")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = SentenceTransformerTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        learning_rate=LR,
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        dataloader_num_workers=0,          # Windows: 0 voorkomt multiprocessing-gedoe
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{RECALL_KEY}",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    trainer.train()

    final = evaluator(model)   # model = beste checkpoint (load_best_model_at_end)
    print(f"\nBASELINE   Recall@10: {base.get(RECALL_KEY, float('nan')):.3f}")
    print(f"FINE-TUNED Recall@10: {final.get(RECALL_KEY, float('nan')):.3f}")

    model.save_pretrained(OUT_DIR)
    print("opgeslagen in", OUT_DIR)


if __name__ == "__main__":
    main()
