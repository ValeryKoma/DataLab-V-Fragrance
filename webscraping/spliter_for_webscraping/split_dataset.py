import pandas as pd
import os

df = pd.read_csv("fragrantica_data/perfumes_table.csv")
missing = df[df["rating"].isna()].copy().reset_index(drop=True)

n = len(missing)
size = n // 3

parts = [
    missing.iloc[:size],
    missing.iloc[size:size * 2],
    missing.iloc[size * 2:],
]

os.makedirs("fragrantica_data", exist_ok=True)

for i, part in enumerate(parts, start=1):
    path = f"fragrantica_data/missing_part{i}.csv"
    part.to_csv(path, index=False)
    print(f"Part {i}: {len(part)} rows → {path}")

print(f"\nTotal missing rows split: {n}")
print("Send each colleague their file + webscrape_missing_parfumo_data.py")
print("They run:  python webscrape_missing_parfumo_data.py --input missing_part1.csv  (adjust number)")
