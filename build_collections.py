"""
Bouwt de split-collecties:
  - fragrances_notes   : 1 doc per parfum  ("title. Brand: ... Notes: ...")
  - fragrances_reviews : 1 doc per recensie (parfum-url in de metadata)

Catalogus komt uit de bestaande fragrances_ft-collectie (zodat ids/urls gelijk blijven);
ruwe reviews joinen we op url uit de Fragrantica-CSV's. Run: python build_collections.py
"""
from __future__ import annotations

import ast
import html
import re
from collections import defaultdict

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# config
CHROMA_PATH  = "./chroma_db"
EMBED_MODEL = "./models/bge-fragrance-v2"   # zelfde model als app.py; na fine-tune hierop wijzen
SOURCE_COLL  = "fragrances_ft"            # canonieke catalogus uit de notebook
NOTES_COLL   = "fragrances_notes"
REVIEWS_COLL = "fragrances_reviews"

MAX_REVIEWS_PER_PERFUME = 6     # langste-eerst; houdt de reviews-collectie behapbaar
MIN_REVIEW_LEN          = 50
DUP_THRESHOLD           = 3     # recensies in >= N parfums = scraper bleed-through -> weg
BATCH                   = 500

REVIEW_FILES = [
    "fragrantica_data/perfumes_table.csv",
    "fragrantica_data/fragnantica_missing_filled.csv",
    "fragrantica_data/missing_part1_filled.csv",
    "fragrantica_data/missing_part2_filled.csv",
    "fragrantica_data/missing_part3_filled.csv",
]

# review-cleaning (zoals notebook cel 8)
_URL_RE  = re.compile(r"https?://\S+")
_SHOW_RE = re.compile(r"\b(?:show|read)\s+more\b", re.I)
_REP_RE  = re.compile(r"([!?.,])\1{2,}")
_WS_RE   = re.compile(r"\s+")


def clean_review(text) -> str:
    if not text:
        return ""
    s = html.unescape(str(text))
    s = _URL_RE.sub(" ", s)
    s = _SHOW_RE.sub(" ", s)
    s = _REP_RE.sub(r"\1\1", s)
    return _WS_RE.sub(" ", s).strip()


def raw_reviews(raw) -> list[str]:
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if s.startswith("["):
        try:
            return [str(r) for r in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return [s]
    return [s] if s else []


# build
def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    src = client.get_collection(SOURCE_COLL)
    n = src.count()
    print(f"Bron {SOURCE_COLL}: {n:,} parfums")

    # 1) catalogus uit fragrances_ft (gepagineerd; alles in 1x knalt tegen de SQLite-limiet)
    cat = []
    for off in range(0, n, BATCH):
        g = src.get(include=["metadatas"], limit=BATCH, offset=off)
        cat.extend({"id": cid, **m} for cid, m in zip(g["ids"], g["metadatas"]))
    url2meta = {c["url"]: c for c in cat}
    print(f"catalogus geladen: {len(cat):,}")

    # 2) ruwe reviews uit de CSV's, gejoined op url (eerste niet-lege rij per url wint)
    raw_by_url: dict[str, list[str]] = {}
    for fp in REVIEW_FILES:
        try:
            dfp = pd.read_csv(fp, usecols=["url", "reviews"])
        except (FileNotFoundError, ValueError):
            print("  overslaan (niet gevonden / geen reviews-kolom):", fp)
            continue
        for u, rv in zip(dfp["url"], dfp["reviews"]):
            if u in url2meta and u not in raw_by_url:
                lst = raw_reviews(rv)
                if lst:
                    raw_by_url[u] = lst
    print(f"parfums met ruwe reviews: {len(raw_by_url):,}")

    # cross-perfume duplicate reviews (scraper bleed-through)
    counts: dict[str, set] = defaultdict(set)
    for u, lst in raw_by_url.items():
        for r in lst:
            c = clean_review(r).lower()
            if len(c) >= MIN_REVIEW_LEN:
                counts[c].add(u)
    dup = {k for k, us in counts.items() if len(us) >= DUP_THRESHOLD}
    print(f"cross-perfume dup reviews verwijderd: {len(dup):,}")

    def perfume_reviews(u: str) -> list[str]:
        out = []
        for r in raw_by_url.get(u, []):
            c = clean_review(r)
            if len(c) >= MIN_REVIEW_LEN and c.lower() not in dup:
                out.append(c)
        out.sort(key=len, reverse=True)
        return out[:MAX_REVIEWS_PER_PERFUME]

    n_rev_by_url = {u: len(perfume_reviews(u)) for u in url2meta}
    print(f"parfums met bruikbare reviews: {sum(v > 0 for v in n_rev_by_url.values()):,}")

    existing = [c.name for c in client.list_collections()]

    # 3) fragrances_notes - 1 doc per parfum (alle 85k)
    if NOTES_COLL in existing:
        print(f"{NOTES_COLL}: bestaat al ({client.get_collection(NOTES_COLL).count():,}) - overslaan")
    else:
        nc = client.create_collection(NOTES_COLL, metadata={"hnsw:space": "cosine"})
        for s in range(0, len(cat), BATCH):
            chunk = cat[s:s + BATCH]
            texts = [f"{c.get('title','')}. Brand: {c.get('brand','')}. Notes: {c.get('notes','')}."
                     for c in chunk]
            embs = model.encode(texts, batch_size=64, show_progress_bar=False).tolist()
            metas = [{
                "title": c.get("title", ""), "brand": c.get("brand", ""),
                "gender": c.get("gender", ""), "rating": float(c.get("rating") or 0.0),
                "notes": c.get("notes", ""), "url": c["url"],
                "n_reviews": n_rev_by_url.get(c["url"], 0),
            } for c in chunk]
            nc.add(ids=[c["id"] for c in chunk], embeddings=embs, documents=texts, metadatas=metas)
            if (s // BATCH) % 20 == 0:
                print(f"  notes: {min(s + BATCH, len(cat)):,}/{len(cat):,}")
        print(f"{NOTES_COLL}: klaar ({nc.count():,} docs)")

    # 4) fragrances_reviews - 1 doc per recensie
    if REVIEWS_COLL in existing:
        print(f"{REVIEWS_COLL}: bestaat al ({client.get_collection(REVIEWS_COLL).count():,}) - overslaan")
    else:
        rc = client.create_collection(REVIEWS_COLL, metadata={"hnsw:space": "cosine"})
        bids, btxt, bmeta = [], [], []

        def flush() -> None:
            if not btxt:
                return
            embs = model.encode(btxt, batch_size=64, show_progress_bar=False).tolist()
            rc.add(ids=bids, embeddings=embs, documents=btxt, metadatas=bmeta)
            bids.clear(); btxt.clear(); bmeta.clear()

        for u, m in url2meta.items():
            for j, rv in enumerate(perfume_reviews(u)):
                bids.append(f"{m['id']}-{j}")
                btxt.append(rv)                       # recensie = document (geen query-prefix)
                bmeta.append({"url": u, "gender": m.get("gender", ""),
                              "title": m.get("title", ""), "brand": m.get("brand", "")})
                if len(btxt) >= 1000:
                    flush()
                    if rc.count() % 20000 < 1000:
                        print(f"  reviews: {rc.count():,} docs...")
        flush()
        print(f"{REVIEWS_COLL}: klaar ({rc.count():,} review-docs)")


if __name__ == "__main__":
    main()
