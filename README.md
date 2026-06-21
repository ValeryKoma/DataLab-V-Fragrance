# Fragrance Advisor

A conversational perfume recommender: hybrid retrieval (notes + reviews) over ChromaDB
with a fine-tuned BGE embedder, plus an LLM (local **Ollama** or **OpenAI**) for the chat
and the written recommendations.

> **Heads-up:** the embedding model, the vector database and the source CSVs are **not in
> this repo** — they're multiple GB, far over GitHub's 100 MB limit. You download them
> separately (link below). You do **not** need a GPU and you do **not** need to fine-tune.

---

## What you actually have to run

| Script | Run it? |
|---|---|
| `app.py` | **Yes** — this is the web app you start. |
| `recommender.py` | No — it's a module `app.py` imports. |
| `build_collections.py` | Only if you rebuild the DB yourself (you don't have to). |
| `finetune_embeddings.py` | No — GPU-only, ~1 h. The shared model already includes it. |

---

## Setup (the easy path — no GPU, no rebuild)

### 1. Clone + install
```bash
git clone https://github.com/ckires/DataLab-V-Fragrance.git
cd DataLab-V-Fragrance
python -m venv .venv && .venv\Scripts\activate      # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### 2. Download the artifacts (model + database)
They're not in git. Download the zip here: **<PASTE-YOUR-SHARE-LINK-HERE>**

Unzip it into the repo root so you end up with:
```
DataLab-V-Fragrance/
├─ models/bge-fragrance-v2/      <- the fine-tuned embedder
├─ chroma_db/                    <- the prebuilt vector database
├─ app.py
└─ ...
```
That's it — no fine-tuning, no `build_collections.py`. The DB already has all ~85k perfumes embedded.

### 3. Choose your LLM backend (edit `app.py`)
**Option A — Ollama (free, local, no API key):**
```python
LLM_BACKEND = "ollama"
```
Then install [Ollama](https://ollama.com/download) and pull the model once:
```bash
ollama pull qwen2.5:7b
```

**Option B — OpenAI (needs your own API key, costs money):**
```python
LLM_BACKEND = "openai"
```
Copy the config template and add your key:
```bash
copy config.example.py config.py      # macOS/Linux: cp config.example.py config.py
```
Put your key in `config.py`, and make sure the `"openai"` model id in `app.py` is a real
model your key can use (e.g. `gpt-4o-mini`).

### 4. Run
```bash
python app.py
```
Open **http://127.0.0.1:8000**. On startup it should print `retrieval: split (notes+reviews)`.

---

## FAQ

- **Do I need a GPU?** No (only `finetune_embeddings.py` does, and you don't run it).
- **The startup says `terugval op fragrances_ft` / dimension error?** Your `chroma_db/` didn't
  unzip into the repo root, or it doesn't match the model. Re-check step 2.
- **`from config import ...` error?** You don't have `config.py`; it's optional now — only
  needed for OpenAI. Ollama works without it.

---

## Advanced: rebuild everything yourself

Only if you didn't download the artifacts:

1. Get the source CSVs (shared separately) into `fragrantica_data/` and `luckyscent_data/`.
2. Build the vector DB: `python build_collections.py`
   (point `EMBED_MODEL` at `BAAI/bge-base-en-v1.5` to auto-download a base model, or at the
   shared `./models/bge-fragrance-v2` for the fine-tuned one — they must match the DB).
3. *(Optional, GPU)* Re-train the embedder: `python finetune_embeddings.py`, then rebuild the DB.
