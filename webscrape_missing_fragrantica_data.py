import time
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup

CHECKPOINT_PATH = "fragrantica_data/fragnantica_missing_filled.csv"
CHECKPOINT_EVERY = 100

# Laad de originele parfumtabel in als dataframe
df = pd.read_csv("fragrantica_data/perfumes_table.csv")

# Filter alle rijen waar de rating ontbreekt
df_missing_rating = df[df["rating"].isna()].copy()

# Maak een kopie van de ontbrekende rijen en voeg nieuwe kolommen toe
df_filled_rating = df_missing_rating.copy()
df_filled_rating[["rating", "best_rating", "rating_count", "normalized_rating", "description", "reviews"]] = None

print(f"Total rows to scrape: {len(df_filled_rating)}")

# Start een sessie die een echte Edge-browser nabootst om Cloudflare te omlopen
session = requests.Session(impersonate="edge101")

for i, (idx, row) in enumerate(df_filled_rating.iterrows(), start=1):
    try:
        # Haal de HTML op
        response = session.get(row["url"], timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # Zoek de rating
        rating_el    = soup.find("span", itemprop="ratingValue")
        best_el      = soup.find("span", itemprop="bestRating")
        count_el     = soup.find("span", itemprop="ratingCount")

        # datatype of None als het element niet bestaat
        rating_value = float(rating_el.text.strip()) if rating_el else None
        best_rating  = float(best_el.text.strip())   if best_el  else None
        rating_count = int(count_el.text.strip())    if count_el else None

        # Sla de rating op 
        df_filled_rating.at[idx, "rating"]            = rating_value
        df_filled_rating.at[idx, "best_rating"]       = best_rating
        df_filled_rating.at[idx, "rating_count"]      = rating_count
        df_filled_rating.at[idx, "normalized_rating"] = round(rating_value / best_rating, 4) if rating_value and best_rating else None

        # Haal de beschrijving op als die nog ontbreekt in de originele data
        if pd.isna(row["description"]):
            desc_el = soup.find("div", id="perfume-description-content")
            df_filled_rating.at[idx, "description"] = desc_el.get_text(separator=" ", strip=True) if desc_el else None

        # Haal de reviews op als die nog ontbreken in de originele data
        if pd.isna(row["reviews"]) or row["reviews"] in ("[]", ""):
            df_filled_rating.at[idx, "reviews"] = [el.get_text(separator=" ", strip=True) for el in soup.find_all("div", itemprop="reviewBody")]

        print(f"[{i}/{len(df_filled_rating)}] {row['title'][:50]} → {rating_value}/{best_rating} ({rating_count} votes) | {row['url']}")

    except Exception as e:
        print(f"[{i}/{len(df_filled_rating)}] {row['title'][:50]} → FAILED ({e}) | {row['url']}")

    # backup elke 100
    if i % CHECKPOINT_EVERY == 0:
        df_filled_rating.to_csv(CHECKPOINT_PATH, index=False)
        print(f"  checkpoint saved ({i} rows done)")

    # antiblock
    time.sleep(2)

df_filled_rating.to_csv(CHECKPOINT_PATH, index=False)
print(f"Done. Saved {CHECKPOINT_PATH}")
