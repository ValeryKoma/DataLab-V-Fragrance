"""
Gedeelde retrieval-core (zowel de notebook als app.py gebruiken deze Recommender).

Hybride, gepersonaliseerde pipeline:
  query -> embedding + smaakprofiel (Rocchio) -> notes- + reviews- + BM25-pass
  -> RRF-fusie -> lichte rating-nudge -> top-k.
Catalogus en BM25-index komen uit de notes-collectie; een reviews-collectie is optioneel.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------- query-expansion
VIBE_HINTS = {
    "date":    "evening seductive sensual warm intimate",
    "office":  "clean professional subtle fresh daytime",
    "summer":  "fresh citrus light aquatic bright",
    "winter":  "warm amber spicy oud heavy cozy",
    "gym":     "fresh sporty clean light aquatic",
    "wedding": "elegant sophisticated floral refined",
    "spicy":   "cinnamon pepper saffron cardamom clove ginger warm oriental",
    "sweet":   "vanilla caramel honey tonka gourmand",
    "woody":   "cedar sandalwood oud patchouli vetiver",
    "fresh":   "citrus bergamot lemon aquatic green",
    "floral":  "rose jasmine lily peony iris",
}

# bge-modellen verwachten een korte instructie vóór de QUERY (niet vóór documenten).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def expand_query(q: str) -> str:
    extras = [hint for kw, hint in VIBE_HINTS.items() if kw in q.lower()]
    return q if not extras else f"{q}. Related notes and vibe: {' '.join(extras)}"


# ---------------------------------------------------------------- negatie / exclusions
# Harde uitsluiting hoort niet in de embedding ("geen zoet" trekt juist naar zoet);
# we parsen negatie uit de tekst en filteren de kandidaten na retrieval.

# Categoriewoord -> note-/merk-roots. Prefix-match (len >= 4): 'vanil' vangt ook
# 'vanilla'/'vanille' (en spelfouten in de data).
EXCLUDE_SYNONYMS = {
    "sweet":   ["sweet", "sugar", "vanil", "caramel", "honey", "tonka", "gourmand",
                "candy", "marshmallow", "chocolat", "cocoa", "praline", "toffee",
                "butterscotch", "syrup", "cream", "milk"],
    "spicy":   ["spic", "cinnamon", "pepper", "saffron", "cardamom", "clove",
                "ginger", "nutmeg"],
    "woody":   ["wood", "cedar", "sandalwood", "oud", "agarwood", "patchouli", "vetiver"],
    "fresh":   ["fresh", "citrus", "bergamot", "lemon", "lime", "aquatic", "marine", "green"],
    "floral":  ["floral", "flower", "rose", "jasmine", "lily", "peony", "iris",
                "violet", "tuberose"],
    "fruity":  ["fruit", "peach", "apple", "berr", "raspberr", "pear", "mango",
                "cherr", "strawberr"],
    "musky":   ["musk"],
    "powdery": ["powder", "orris"],
    "smoky":   ["smok", "incense", "tobacco", "leather", "birch", "tar"],
    "animalic": ["animalic", "civet", "castoreum", "oud", "leather"],
}

# negatie-cues: woorden die aangeven dat het volgende vermeden moet worden
_NEG_CUE = (r"(?:no|not|nothing|none|without|avoid(?:ing)?|hates?|dislikes?|anti|"
            r"don'?t\s+(?:like|want)|do\s+not\s+(?:like|want)|can'?t\s+stand)")
# een term mag zelf geen cue-woord zijn, zodat "no sweet and no fruity" beide vangt
_CUE_WORDS = (r"(?:no|not|nothing|none|without|avoid|avoiding|hates?|dislikes?|anti|"
              r"and|or)")
_NEG_TERM = rf"(?!{_CUE_WORDS}\b)[a-z][a-z']+"
# vulwoorden tussen cue en term die we overslaan ("no anything from cb" -> 'cb')
_FILLER = (r"(?:too|so|very|any|anything|much|a\s+lot\s+of|from|of|the|some|something|"
           r"with|containing|that\s+(?:has|have|contains?)|has|have|contains?|"
           r"perfumes?|scents?|fragrances?|colognes?|smells?|notes?|ones?|things?|an?)\s+")
_NEG_RE = re.compile(
    rf"\b{_NEG_CUE}\b[\s,]*(?:{_FILLER})*"
    rf"(?P<terms>{_NEG_TERM}(?:\s*(?:,|and|or|&)\s*{_NEG_TERM}){{0,3}})",
    re.I,
)
_CONNECTOR_RE = re.compile(r"\s*(?:,|\band\b|\bor\b|&)\s*", re.I)


def parse_negations(text: str) -> tuple[str, list[str]]:
    """Splitst vrije tekst in (schone_tekst, exclude-termen), bv.
    'office, nothing sweet' -> ('office', ['sweet']). De negatie-clausule wordt uit de
    tekst gehaald zodat de embedding niet alsnog naar de uitgesloten note drijft."""
    excludes: list[str] = []

    def _grab(m: "re.Match") -> str:
        for t in _CONNECTOR_RE.split(m.group("terms")):
            t = t.lower().strip()
            if t and t not in ("and", "or"):
                excludes.append(t)
        return " "

    clean = _NEG_RE.sub(_grab, text)
    clean = re.sub(r"\s*,(?:\s*,)+", ", ", clean)   # ", ," -> ", "
    clean = re.sub(r"\s+", " ", clean).strip(" ,")
    # dedup met behoud van volgorde
    seen: set[str] = set()
    excludes = [t for t in excludes if not (t in seen or seen.add(t))]
    return clean, excludes


def expand_excludes(terms: list[str] | None) -> set[str]:
    """Categoriewoord -> set van te filteren roots (via EXCLUDE_SYNONYMS).
    Onbekende termen (bv. een merk-afkorting als 'cb') blijven zichzelf."""
    out: set[str] = set()
    for t in terms or []:
        t = t.lower().strip()
        if t:
            out.add(t)
            out.update(EXCLUDE_SYNONYMS.get(t, []))
    return out


def excluded_by(text: str, excl: set[str]) -> bool:
    """True als een uitgesloten root in `text` voorkomt (prefix-match len >= 4, korte
    termen exact). `text` = notes + title + brand, zodat ook merk/naam wordt afgevangen."""
    if not excl:
        return False
    for tok in tokenize(text):
        for e in excl:
            if tok == e or (len(e) >= 4 and tok.startswith(e)):
                return True
    return False


# ---------------------------------------------------------------- recommender
class Recommender:
    """Hybride, gepersonaliseerde recommender bovenop een ChromaDB-collectie."""

    def __init__(self, collection, model, reviews_collection=None, page: int = 500,
                 query_prefix: str = BGE_QUERY_PREFIX):
        self.collection = collection                  # notes: 1 doc per parfum (alle ~85k)
        self.reviews_collection = reviews_collection  # 1 doc per recensie (optioneel)
        self.n_review_docs = reviews_collection.count() if reviews_collection is not None else 0
        self.model = model
        self.query_prefix = query_prefix   # "" zetten voor niet-bge modellen

        # catalogus gepagineerd inladen (alles ineens raakt de SQLite-limiet)
        cat = []
        n = collection.count()
        for off in range(0, n, page):
            g = collection.get(include=["metadatas"], limit=page, offset=off)
            cat.extend({"id": cid, **m} for cid, m in zip(g["ids"], g["metadatas"]))
        self.catalog = cat
        self.id2meta = {c["id"]: c for c in cat}
        self.url2meta = {c["url"]: c for c in cat}
        # heeft dit parfum reviews? (uit notes-metadata n_reviews) -> voor de fairness-nudge
        self.url_has_reviews = {c["url"]: bool(c.get("n_reviews", 0)) for c in cat}

        rts = [c["rating"] for c in cat
               if isinstance(c.get("rating"), (int, float)) and c["rating"] > 0]
        self.rating_mean = sum(rts) / len(rts) if rts else 3.5

        # BM25 over de lexicale velden (notes + title + brand)
        self.bm25 = BM25Okapi([
            tokenize(f"{c.get('notes', '')} {c.get('title', '')} {c.get('brand', '')}")
            for c in cat
        ])

        # vocabulaire van alle note-woorden (voor de faithfulness-/hallucinatie-check)
        self.note_vocab = set()
        for c in cat:
            self.note_vocab.update(tokenize(c.get("notes", "")))

    # -------------------------------------------------- smaakprofiel
    def profile_vector(self, profile: list[dict]):
        """Rocchio-smaakvector uit (id, cijfer): gewicht = cijfer - 3 (laag stoot af,
        hoog trekt aan). Embeddings genormaliseerd zodat alleen de richting telt."""
        if not profile:
            return None
        got = self.collection.get(ids=[p["id"] for p in profile], include=["embeddings"])
        id2emb = dict(zip(got["ids"], np.asarray(got["embeddings"], dtype=float)))
        vecs, weights = [], []
        for p in profile:
            e = id2emb.get(p["id"])
            w = float(p["rating"]) - 3.0
            if e is None or w == 0:
                continue
            vecs.append(e / (np.linalg.norm(e) + 1e-9))
            weights.append(w)
        if not vecs:
            return None
        v = (np.asarray(vecs) * np.asarray(weights).reshape(-1, 1)).sum(axis=0)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 1e-6 else None   # None bij netto-nul (likes/dislikes heffen op)

    # -------------------------------------------------- retrieval
    def recommend(self, query: str = "", profile: list[dict] | None = None,
                  gender: str | None = None, top_k: int = 5, min_rating: float = 0.0,
                  exclude_notes: list[str] | None = None,
                  alpha: float = 0.5, rating_weight: float = 0.15, use_bm25: bool = True,
                  reviews: bool = True, review_weight: float = 1.0, review_fairness: float = 1.0,
                  expand: bool = True, rrf_k: int = 60, pool: int = 200) -> pd.DataFrame:
        """
        query         : vrije tekst
        profile       : [{'id','rating'}] smaakprofiel; None = onpersoonlijk
        gender        : 'men'|'women'|'unisex'|None
        exclude_notes : harde uitsluiting (notes/naam/merk), via EXCLUDE_SYNONYMS
        alpha         : gewicht query-tekst t.o.v. smaakprofiel (0..1)
        rating_weight : lichte community-rating-nudge (0 = uit)
        use_bm25      : hybride (vector + BM25) of puur vector
        reviews       : reviews-collectie meenemen als die er is
        review_weight : gewicht reviews-pass in de RRF
        review_fairness: multiplier (>1) voor parfums zonder reviews; 1.0 = uit
        expand        : query-expansion aan/uit
        """
        profile = profile or []
        excl = expand_excludes(exclude_notes)
        text = expand_query(query) if (expand and query) else query

        # Zoekvector = blend van query-embedding en smaakvector
        cv = None
        if text.strip():
            cv = np.asarray(self.model.encode([self.query_prefix + text])[0], dtype=float)
            cv = cv / (np.linalg.norm(cv) + 1e-9)
        pv = self.profile_vector(profile)
        if cv is not None and pv is not None:
            qv = alpha * cv + (1 - alpha) * pv
        else:
            qv = cv if cv is not None else pv
        if qv is None:
            return pd.DataFrame()
        qv = (qv / (np.linalg.norm(qv) + 1e-9)).tolist()

        where = {"gender": {"$eq": gender}} if gender in ("men", "women", "unisex") else None

        # 1) notes vector-pass (alle parfums, 1 doc elk)
        vres = self.collection.query(query_embeddings=[qv], n_results=pool,
                                     where=where, include=["metadatas"])
        vec_urls = [m["url"] for m in vres["metadatas"][0]]

        # 2) reviews vector-pass (1 doc per recensie) -> map terug naar parfum, beste rang telt
        review_urls = []
        if reviews and self.reviews_collection is not None and self.n_review_docs:
            n_rev = min(pool * 2, self.n_review_docs)
            rres = self.reviews_collection.query(query_embeddings=[qv], n_results=n_rev,
                                                 where=where, include=["metadatas"])
            seen_r = set()
            for m in rres["metadatas"][0]:
                u = m.get("url")
                if u and u not in seen_r:        # eerste (= best gerangschikte) recensie wint
                    seen_r.add(u)
                    review_urls.append(u)

        # 3) BM25-pass op de tekst (lexicaal; de smaak zit al in de vector)
        bm_urls = []
        if use_bm25 and text.strip():
            scores = self.bm25.get_scores(tokenize(text))
            for i in np.argsort(scores)[::-1]:
                if scores[i] <= 0:
                    break
                m = self.catalog[i]
                if where and m.get("gender") != gender:
                    continue
                bm_urls.append(m["url"])
                if len(bm_urls) >= pool:
                    break

        # 4) Reciprocal Rank Fusion. notes + BM25 dekken alles, reviews maar een deel;
        # review_fairness tilt parfums zonder reviews op zodat ze niet structureel zakken.
        rrf = defaultdict(float)
        for r, u in enumerate(vec_urls):
            rrf[u] += 1.0 / (rrf_k + r)
        for r, u in enumerate(review_urls):
            rrf[u] += review_weight / (rrf_k + r)
        for r, u in enumerate(bm_urls):
            rrf[u] += 1.0 / (rrf_k + r)
        if not rrf:
            return pd.DataFrame()
        if review_fairness != 1.0:
            for u in rrf:
                if not self.url_has_reviews.get(u, False):
                    rrf[u] *= review_fairness
        max_rrf = max(rrf.values())

        # 4) Filteren + lichte rating-nudge + eigen keuzes uitsluiten
        seed_urls = {self.id2meta[p["id"]]["url"] for p in profile if p["id"] in self.id2meta}
        rows = []
        for u, s in rrf.items():
            if u in seed_urls:
                continue
            m = self.url2meta.get(u)
            if m is None:
                continue
            if excl and excluded_by(f"{m.get('notes','')} {m.get('title','')} {m.get('brand','')}", excl):
                continue   # harde constraint: uitgesloten note/merk/naam -> kandidaat valt af
            rating = float(m.get("rating") or 0.0)
            if rating < min_rating:
                continue
            rt = rating if rating > 0 else self.rating_mean
            score = (1 - rating_weight) * (s / max_rrf) + rating_weight * (rt / 5.0)
            rows.append({
                "title": m.get("title"), "brand": m.get("brand"), "gender": m.get("gender"),
                "rating": round(rating, 2), "notes": m.get("notes"), "url": u,
                "score": round(score, 3),
            })
        rows.sort(key=lambda r: r["score"], reverse=True)

        # dedup op (title, brand): zelfde parfum staat soms onder twee URLs
        seen, deduped = set(), []
        for r in rows:
            key = (str(r["title"]).strip().lower(), str(r["brand"]).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
            if len(deduped) >= top_k:
                break
        return pd.DataFrame(deduped)

    def hallucinated_notes(self, generated_text: str, recs: list[dict]) -> list[str]:
        """Faithfulness-check: note-woorden in de LLM-tekst die wél in het globale
        note-vocabulaire zitten maar in GEEN van de aanbevolen parfums voorkomen."""
        allowed = set()
        for m in recs:
            allowed.update(tokenize(m.get("notes", "")))
        return sorted((set(tokenize(generated_text)) & self.note_vocab) - allowed)
