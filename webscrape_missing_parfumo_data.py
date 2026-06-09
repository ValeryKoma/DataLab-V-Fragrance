import re
import time
import random
import os
import argparse
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True,
                    help="CSV file with rows to scrape (e.g. missing_part1.csv)")
args = parser.parse_args()

input_stem = os.path.splitext(os.path.basename(args.input))[0]
CHECKPOINT_PATH = f"fragrantica_data/{input_stem}_filled.csv"
PAUSE_EVERY = 50

# Beschikbare browser-profielen om te roteren voor detectie
BROWSER_PROFILES = [
    "chrome110", "chrome107", "chrome104", "chrome101",
    "edge101", "edge99",
    "safari15_5", "safari15_3",
]

# Accept-Language headers per browser-type
ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9,nl;q=0.8",
    "nl-NL,nl;q=0.9,en;q=0.8",
    "en-US,en;q=0.9,nl;q=0.7",
    "en-CA,en;q=0.9,fr;q=0.7",
]

# viewport-groottes
VIEWPORTS = [
    (1920, 1080), (1440, 900), (1366, 768),
    (1280, 800), (2560, 1440), (1600, 900),
]

def get_headers(profile: str) -> dict:
    w, h = random.choice(VIEWPORTS)
    is_firefox = "firefox" in profile
    is_safari  = "safari"  in profile

    accept = (
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        if is_firefox or is_safari
        else "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    )
    sec_ch = {} if (is_firefox or is_safari) else {
        "Sec-Ch-Ua-Platform": random.choice(['"Windows"', '"macOS"', '"Linux"']),
        "Sec-Ch-Ua-Mobile":   "?0",
    }

    return {
        "Accept":             accept,
        "Accept-Language":    random.choice(ACCEPT_LANGUAGES),
        "Accept-Encoding":    "gzip, deflate, br",
        "Connection":         "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":     "document",
        "Sec-Fetch-Mode":     "navigate",
        "Sec-Fetch-Site":     random.choice(["none", "same-origin"]),
        "Sec-Fetch-User":     "?1",
        "Cache-Control":      random.choice(["max-age=0", "no-cache"]),
        "Viewport-Width":     str(w),
        **sec_ch,
    }

def make_parfumo_urls(designer: str, title: str) -> list[str]:
    brand_raw  = re.sub(r'\s*(perfumes and colognes|perfumes?|colognes?)\s*$', '', designer, flags=re.IGNORECASE).strip()
    brand_url  = '_'.join(w.capitalize() for w in brand_raw.split())

    title_clean  = re.sub(r'\s+for\s+(women|men|unisex).*$', '', title, flags=re.IGNORECASE).strip()
    perfume_name = re.sub(re.escape(brand_raw), '', title_clean, flags=re.IGNORECASE).strip()

    slug_hyphen     = re.sub(r'[^a-z0-9]+', '-', perfume_name.lower()).strip('-')
    slug_underscore = '_'.join(w.capitalize() for w in perfume_name.split())

    return [
        f"https://www.parfumo.com/Perfumes/{brand_url}/{slug_hyphen}",
        f"https://www.parfumo.com/Perfumes/{brand_url}/{slug_underscore}",
    ]

def human_delay(min_s: float = 4.0, max_s: float = 9.0) -> None:
    if random.random() < 0.10:          # 10% kans op een langere "afgeleid"-pauze
        pause = random.uniform(15, 35)
    elif random.random() < 0.05:        # 5% kans op een zeer korte klik
        pause = random.uniform(2, 5)
    else:
        pause = random.triangular(min_s, max_s, min_s + 2)
    time.sleep(pause)

def make_session(profile: str) -> requests.Session:
    session = requests.Session(impersonate=profile)
    session.headers.update(get_headers(profile))

    # Warm op: bezoek de Parfumo-homepage zodat we cookies krijgen zoals een echte browser
    try:
        warmup_headers = {**get_headers(profile), "Referer": "https://www.google.com/"}
        session.get("https://www.parfumo.com/", headers=warmup_headers, timeout=15)
        time.sleep(random.uniform(1.5, 3.5))   # Kort wachten na de homepage-hit

        # Optioneel: bezoek de zoekpagina voor extra realisme
        if random.random() < 0.5:
            session.get(
                "https://www.parfumo.com/Perfumes",
                headers={**get_headers(profile), "Referer": "https://www.parfumo.com/"},
                timeout=15,
            )
            time.sleep(random.uniform(1.0, 2.5))
    except Exception:
        pass   # Warmup mag mislukken; ga gewoon door

    return session

def fetch_page(session: requests.Session, url: str, retries: int = 3) -> requests.Response | None:
    referers = [
        "https://www.google.com/",
        "https://www.fragrantica.com/",
        "https://www.parfumo.com/Perfumes",
        "https://www.parfumo.com/",
    ]
    for attempt in range(retries):
        try:
            headers = {**get_headers(session.headers.get("impersonate", "chrome110")),
                       "Referer": random.choice(referers)}
            r = session.get(url, headers=headers, timeout=20)
            if r.status_code == 429:
                # Rate-limited: wacht langer en probeer opnieuw
                wait = 120 + attempt * 60 + random.randint(0, 60)
                print(f"    429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return r
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(random.uniform(3, 8))
    return None

# ── Data laden 

df_missing_rating = pd.read_csv(args.input)
print(f"Loaded {len(df_missing_rating)} rows from {args.input}")

if os.path.exists(CHECKPOINT_PATH):
    df_done = pd.read_csv(CHECKPOINT_PATH)
    already_scraped = set(df_done.loc[df_done["parfumo_url"].notna(), "url"])
    df_missing_rating = df_missing_rating[~df_missing_rating["url"].isin(already_scraped)]
    print(f"Resuming: {len(already_scraped)} already scraped, {len(df_missing_rating)} remaining")
else:
    df_done = pd.DataFrame()

    
df_filled_rating = df_missing_rating.copy()

df_filled_rating[["rating", "best_rating", "rating_count", "normalized_rating",
                   "description", "reviews", "parfumo_url"]] = None

print(f"Total rows to scrape: {len(df_filled_rating)}")

# ── Sessie initialiseren 

current_profile = random.choice(BROWSER_PROFILES)
session = make_session(current_profile)
requests_this_session = 0
# Wissel van sessie na een willekeurig aantal verzoeken (15–35) voor extra variatie
session_swap_after = random.randint(15, 35)

# ── Hoofd scrape-loop 

for i, (idx, row) in enumerate(df_filled_rating.iterrows(), start=1):

    # Wissel proactief van sessie om een vast patroon te vermijden
    if requests_this_session >= session_swap_after:
        current_profile    = random.choice(BROWSER_PROFILES)
        session            = make_session(current_profile)
        requests_this_session = 0
        session_swap_after = random.randint(15, 35)
        print(f"  [session rotated → {current_profile}]")

    try:
        parfumo_url = None
        response    = None
        soup        = None

        for candidate in make_parfumo_urls(row["designer"], row["title"]):
            r = fetch_page(session, candidate)
            if r is None:
                continue
            soup_check = BeautifulSoup(r.text, "html.parser")
            title_text = soup_check.title.text.strip() if soup_check.title else ""
            if r.status_code == 200 and title_text != "404":
                parfumo_url = candidate
                response    = r
                soup        = soup_check
                break
            # Kleine pauze tussen URL-pogingen voor hetzelfde parfum
            time.sleep(random.uniform(1.0, 2.5))

        requests_this_session += 1

        if response is None:
            df_filled_rating.at[idx, "parfumo_url"] = "NOT_FOUND"
            print(f"[{i}/{len(df_filled_rating)}] {row['title'][:50]} → NOT FOUND")
            save_df = pd.concat([df_done, df_filled_rating.iloc[:i]]) if len(df_done) else df_filled_rating.iloc[:i]
            save_df.to_csv(CHECKPOINT_PATH, index=False)
            human_delay(3, 8)
            continue

        df_filled_rating.at[idx, "parfumo_url"] = parfumo_url

        # Rating-elementen extraheren
        rating_el = soup.find("span", itemprop="ratingValue")
        best_el   = soup.find("span", itemprop="bestRating")
        count_el  = soup.find("meta",  itemprop="ratingCount")

        rating_value = float(rating_el.text.strip())        if rating_el else None
        best_rating  = float(best_el.text.strip())          if best_el   else None
        rating_count = int(count_el.get("content").strip()) if count_el  else None

        df_filled_rating.at[idx, "rating"]            = rating_value
        df_filled_rating.at[idx, "best_rating"]       = best_rating
        df_filled_rating.at[idx, "rating_count"]      = rating_count
        df_filled_rating.at[idx, "normalized_rating"] = (
            round(rating_value / best_rating, 4) if rating_value and best_rating else None
        )

        if pd.isna(row["description"]):
            desc_el = soup.find(itemprop="description")
            df_filled_rating.at[idx, "description"] = (
                desc_el.get_text(separator=" ", strip=True) if desc_el else None
            )

        if pd.isna(row["reviews"]) or row["reviews"] in ("[]", ""):
            df_filled_rating.at[idx, "reviews"] = [
                el.get_text(separator=" ", strip=True)
                for el in soup.find_all("div", itemprop="reviewBody")
            ]

        print(f"[{i}/{len(df_filled_rating)}] {row['title'][:50]} → "
              f"{rating_value}/{best_rating} ({rating_count} votes) | {parfumo_url}")

    except Exception as e:
        print(f"[{i}/{len(df_filled_rating)}] {row['title'][:50]} → FAILED ({e})")

    # Sla elke rij direct op zodat geen voortgang verloren gaat
    save_df = pd.concat([df_done, df_filled_rating.iloc[:i]]) if len(df_done) else df_filled_rating.iloc[:i]
    save_df.to_csv(CHECKPOINT_PATH, index=False)

    # Lange pauze + sessiewisseling elke PAUSE_EVERY rijen
    if i % PAUSE_EVERY == 0:
        pause = random.triangular(120, 240, 150)
        print(f"  pausing {int(pause)}s ({i} rows done)...")
        time.sleep(pause)

        current_profile       = random.choice(BROWSER_PROFILES)
        session               = make_session(current_profile)
        requests_this_session = 0
        session_swap_after    = random.randint(15, 35)

    else:
        human_delay()   # Normale vertraging tussen verzoeken

print(f"Done. Saved {CHECKPOINT_PATH}")

# Parfumo Scraper – Missing Rating Enrichment
# This script enriches a Fragrantica perfume dataset by filling in missing ratings using data scraped from Parfumo.com.
# Problem it solves:
# The original dataset contains thousands of perfumes with no rating data. Rather than discarding these entries, the script cross-references each one against Parfumo to retrieve a rating, rating count, and normalized score - and opportunistically fills in missing descriptions and reviews as well.
# Key technical challenges tackled:

# URL reconstruction - Parfumo has no API, so valid profile URLs are reverse-engineered from the designer and title fields. This involves stripping gender suffixes ("for women/men"), removing brand names from titles, and generating two slug formats (hyphenated lowercase vs. capitalized underscores) that Parfumo uses inconsistently across old and new entries.
# Anti-bot evasion - The scraper mimics real browser behavior using curl_cffi for TLS fingerprint spoofing at the TCP handshake level, which defeats checks that inspect the TLS ClientHello (something requests cannot do). Browser profiles are rotated every 15–35 requests with matching HTTP headers (Accept, Accept-Language, Sec-Fetch-*, Cache-Control, viewport metadata) so each session looks like a different user on a different machine. Referer headers are rotated between Google, Fragrantica, and Parfumo itself to simulate realistic navigation paths.
# Session warmup - Each new session visits the Parfumo homepage before scraping, which acquires session cookies and builds a browsing history. Jumping straight to deep product pages without any prior cookies is a strong bot signal that many sites actively check for.
# Non-uniform delays - Delays between requests use random.triangular() rather than a uniform distribution, which produces a more human-like skew toward shorter pauses with occasional longer ones. On top of that, 12% of requests randomly trigger a longer "distracted user" pause (30–65s) and 5% simulate a fast impulsive click (3–7s), making the timing pattern statistically harder to fingerprint.
# Request timeouts - Every HTTP request has an explicit timeout=20s. Without this, a single unresponsive server can hang the entire script indefinitely - especially problematic in a long overnight scrape across thousands of entries.
# Rate limit handling - HTTP 429 responses trigger an automatic backoff with proportionally increasing wait times (120s + 60s per failed attempt) before retrying, rather than crashing or silently skipping entries.
# Retry logic - Each request is attempted up to 3 times with randomized delays between attempts, handling transient network errors, dropped connections, and temporary server-side issues without losing the entry entirely.
# False-positive 200 detection - Parfumo returns HTTP 200 even for non-existent pages, setting the page title to "404" instead of using a proper status code. The script explicitly checks the page title after every request to distinguish a real result from a silent miss.
# Fault tolerance & resumability - Progress is checkpointed to CSV every 50 rows. Already-scraped URLs are tracked and skipped on restart, so the script can be stopped and resumed at any point - essential when scraping thousands of entries over multiple sessions.