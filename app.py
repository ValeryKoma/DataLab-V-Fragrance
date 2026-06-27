"""
Fragrance Advisor - FastAPI backend.

Weblaag bovenop Recommender (recommender.py): zoeken, het advisor-gesprek en de
frontend. LLM-backend wisselbaar via LLM_BACKEND ("ollama" of "openai").
Run: python app.py  ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import re

try:
    from config import OPENAI_API_KEY     # lokaal bestand, niet in git (zie config.example.py)
except ImportError:
    OPENAI_API_KEY = ""                    # geen config.py -> val terug op env var / Ollama

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from recommender import Recommender, parse_negations

# config
CHROMA_PATH        = "./chroma_db"
EMBED_MODEL        = "./models/bge-fragrance-v2"
NOTES_COLLECTION   = 'fragrances_notes'     # split: 1 doc per parfum (alle ~85k)
REVIEWS_COLLECTION = 'fragrances_reviews'   # split: 1 doc per recensie
FALLBACK_COLLECTION = 'fragrances_ft'       # oude blob-collectie, tot build_collections.py draaide
ALPHA           = 0.5     # gesprek vs. smaakprofiel
RATING_WEIGHT   = 0.15    # lichte community-rating-nudge

# --- LLM-backend: wissel lokaal <-> OpenAI met deze regel 
LLM_BACKEND = "ollama"            # "ollama" (lokaal, gratis) | "openai"

LLM_BACKENDS = {
    "ollama": {"base_url": "http://localhost:11434/v1",
               "api_key":  "ollama",                       # genegeerd door Ollama
               "model":    "qwen2.5:7b"},
    "openai": {"base_url": None,                            # default OpenAI-endpoint
               "api_key":  OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"),
               "model":    "gpt-5.4"},
}
_LLM      = LLM_BACKENDS[LLM_BACKEND]
LLM_MODEL = _LLM["model"]

# startup
# relative paden (./chroma_db, ./models, static) werken ongeacht van waar je start
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Embedding-model laden...")
model = SentenceTransformer(EMBED_MODEL)

print("ChromaDB verbinden + Recommender bouwen...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = client.get_collection(NOTES_COLLECTION)
except Exception:
    collection = client.get_collection(FALLBACK_COLLECTION)
    print(f"Let op: '{NOTES_COLLECTION}' niet gevonden - terugval op '{FALLBACK_COLLECTION}'. "
          "Draai build_collections.py voor de split.")
try:
    reviews_collection = client.get_collection(REVIEWS_COLLECTION)
except Exception:
    reviews_collection = None
    print(f"Let op: '{REVIEWS_COLLECTION}' niet gevonden - notes-only retrieval.")
rec = Recommender(collection, model, reviews_collection=reviews_collection)
_src = "split (notes+reviews)" if reviews_collection is not None else "notes-only"
print(f"Klaar: {len(rec.catalog):,} parfums | retrieval: {_src}.")

llm = OpenAI(base_url=_LLM["base_url"], api_key=_LLM["api_key"] or "missing")  # geen lege key -> client crasht niet; call faalt netjes
print(f"LLM-backend: {LLM_BACKEND} ({LLM_MODEL})")

ADVISOR_SYS = (
    "You are a friendly fragrance advisor running the intake BEFORE a recommendation. "
    "The user has already rated a few perfumes they like (that taste profile is used "
    "automatically for retrieval). "
    "You do NOT have access to the perfume catalog, so you must NEVER name, list, invent or "
    "describe specific perfumes or brands - any perfume you mention would be a hallucination. "
    "Never write a numbered list. Your ONLY two valid outputs are: (a) ONE short clarifying "
    "question about occasion, season, desired intensity/longevity, notes to love or avoid, or "
    "gender; or (b) the token [RECOMMEND] on its own line when you have enough (or the user asks "
    "for picks). Ask at most 3 questions total. Keep every message short and conversational."
)

REC_SYS = (
    "You are an expert, friendly fragrance advisor. Work ONLY from the candidate perfumes "
    "provided - never invent perfumes, brands or notes, and only mention notes that appear in a "
    "candidate's note list. Only recommend candidates that genuinely fit the request; if a candidate "
    "is a weak match, leave it out rather than padding the list. They are ranked by relevance, so "
    "lead with the strongest. For each, connect it to the user's request through its notes (explain "
    "WHY it fits, don't just restate notes). If the user asked to avoid something, never describe a "
    "pick as having that quality. Avoid marketing fluff. "
    "Answer in English: one short intro sentence, then a numbered list - each item the perfume name "
    "in bold followed by one or two sentences."
)

def conversation_query(history: list[dict], profile: list[dict]) -> tuple[str, list[str]]:
    # Alleen likes in de tekstquery; negaties los via parse_negations (harde filter),
    # want een embedding snapt 'vermijd X' niet en trekt juist naar X.
    likes = [f"{rec.id2meta[p['id']]['title']}: {rec.id2meta[p['id']].get('notes', '')}"
             for p in profile if p["id"] in rec.id2meta and p["rating"] >= 4]
    raw_wants = " ".join(m["content"] for m in history if m["role"] == "user")
    wants, excludes = parse_negations(raw_wants)
    head = ("Likes: " + " | ".join(likes) + ". ") if likes else ""
    return (head + "Looking for: " + wants).strip(), excludes


# Advisor mag geen parfums noemen (zou hallucineren); knip een beginnende lijst weg.
_LIST_CUT_RE = re.compile(r"(?:\n\s*\d+[.)]\s)|one short intro|numbered list|here (?:are|is)",
                          re.I)


def interpret_advisor(raw: str) -> tuple[str, bool]:
    """Bepaalt uit de advisor-output of we aanbevelen, en schoont de tekst op:
    [RECOMMEND] -> aanbevelen; anders knip een lijst weg en toon de vraag, of val
    terug op aanbevelen als het model off-script ging."""
    if "[RECOMMEND]" in raw:
        return "", True
    cut = _LIST_CUT_RE.search(raw)
    head = (raw[:cut.start()] if cut else raw).strip()
    if "?" in head:
        return head.split("?", 1)[0].strip() + "?", False   # alleen de eerste vraag
    if not head or cut:
        return "", True                                      # off-script -> grounded
    return head, False


def ground(conv_text: str, recs: list[dict], exclude_notes: list[str] | None = None) -> str:
    cand = "\n".join(
        f"{i}. {m.get('title')} | brand: {m.get('brand')} | {m.get('gender')} | notes: {m.get('notes')}"
        for i, m in enumerate(recs, 1)
    )
    avoid = (f"\n\nThe user wants to AVOID: {', '.join(sorted(set(exclude_notes)))}. "
             "The candidates below are already filtered to respect this - do not describe any "
             "pick as having those qualities.") if exclude_notes else ""
    user = f'User request: "{conv_text}"{avoid}\n\nCandidate perfumes:\n{cand}'
    r = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": REC_SYS}, {"role": "user", "content": user}],
        temperature=0.7,
    )
    return r.choices[0].message.content.strip()


# API
app = FastAPI(title="Fragrance Advisor")


class Pref(BaseModel):
    id: str
    rating: float = 4.0


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    profile: list[Pref] = []
    history: list[Msg] = []
    message: str = ""
    force: bool = False           # 'Geef aanbeveling nu'
    gender: str | None = None
    alpha: float = ALPHA


def _dump(m: BaseModel) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


@app.get("/api/search")
def search(q: str):
    ql = q.lower().strip()
    if len(ql) < 2:
        return []
    out = []
    for c in rec.catalog:
        if ql in str(c.get("title", "")).lower():
            out.append({"id": c["id"], "label": f"{c.get('title')} · {c.get('brand')}"})
            if len(out) >= 20:
                break
    return out


@app.post("/api/chat")
def chat(req: ChatReq):
    history = [_dump(m) for m in req.history]
    if req.message:
        history.append({"role": "user", "content": req.message})

    reply, do_rec = "", req.force
    if not req.force:
        try:
            msgs = [{"role": "system", "content": ADVISOR_SYS}] + history
            reply = llm.chat.completions.create(
                model=LLM_MODEL, messages=msgs, temperature=0.7
            ).choices[0].message.content.strip()
        except Exception as e:
            hint = "Draait Ollama?" if LLM_BACKEND == "ollama" else "Staat OPENAI_API_KEY gezet?"
            err = f"[LLM niet bereikbaar ({LLM_BACKEND}): {type(e).__name__}. {hint}]"
            history.append({"role": "assistant", "content": err})
            return {"reply": err, "recommendations": [], "history": history}
        reply, do_rec = interpret_advisor(reply)

    recs = []
    if do_rec:
        profile = [_dump(p) for p in req.profile]
        conv, exclude_notes = conversation_query(history, profile)
        recs = rec.recommend(conv, profile=profile, gender=req.gender,
                             alpha=req.alpha, rating_weight=RATING_WEIGHT,
                             exclude_notes=exclude_notes).to_dict("records")
        if recs:
            try:
                grounded = ground(conv, recs, exclude_notes)
            except Exception as e:
                grounded = "Aanbevelingen (LLM-tekst niet beschikbaar: %s)." % type(e).__name__
            reply = grounded   # alleen de grounded tekst noemt parfums; advisor-prose niet
        else:
            reply = "Ik kon geen passende parfums vinden."

    history.append({"role": "assistant", "content": reply})
    cards = [{"title": m.get("title"), "brand": m.get("brand"), "gender": m.get("gender"),
              "notes": m.get("notes"), "url": m.get("url")} for m in recs]
    return {"reply": reply, "recommendations": cards, "history": history}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
