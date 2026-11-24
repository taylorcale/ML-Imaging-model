""" Build manifest from images laid out as:
data/<Species>/<Species>_<scleretype>_<number>.png
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(r"C:\Users\st-lab105\Desktop\freshwater_sponge_model_origianal")
OUT_CSV = Path("data/manifest.csv")
SCLERE_TYPES = {"megasclere", "microsclere", "gemmulosclere", "cluster"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# display row
rows, problems = [], []

for path in sorted(RAW_DIR.rglob("*")):
    if path.suffix.lower() not in IMAGE_EXTS:
        continue

    species = path.parent.name # display species in order from the folder 
    parts = path.stem.split("_")

    if len(parts) < 2:
        problems.append((path.name, "expected ...._scleretype_number"))
        continue

    sclere_type = parts[-2].lower() # second to the last piece of the filename
    if sclere_type not in SCLERE_TYPES:
        problems.append((path.name, f"unknown sclere type '{parts[-2]}'"))
        continue 

    rows.append(
        {
            "species": species, 
            "sclere_type": sclere_type,
            "image_number": parts[-1],
            "image_path": str(path),
        }
    )

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"{len(df)} image written to {OUT_CSV}\n")
if not df.empty:
    print(df.groupby(["species", "sclere_type"]).size().unstack(fill_value=0))

if problems:
    print(f"\n{len(problems)} files skipped:")
    for name, reason in problems:
        print(f"   {name}: {reason}")