from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MERP"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MERP DATASET DOWNLOAD
# ---------------------------------------------------------

REPO_ID = "amaai-lab/MERP"

# We only want the annotation/profile data for now.
# We DO NOT download the large .wav audio files yet.
files = [
    "annotations_filtered.parquet",
    "annotations_raw.parquet",
    "rater_profiles.parquet"
]

# Folder where the data will be stored
data_folder = DATA_DIR

print("Downloading MERP data files...\n")

downloaded_files = {}

for filename in files:
    print(f"Downloading {filename}...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=data_folder
    )

    downloaded_files[filename] = path

    print(f"Saved to: {path}\n")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

filtered = pd.read_parquet(
    downloaded_files["annotations_filtered.parquet"]
)

raw = pd.read_parquet(
    downloaded_files["annotations_raw.parquet"]
)

profiles = pd.read_parquet(
    downloaded_files["rater_profiles.parquet"]
)


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\nFILTERED ANNOTATIONS")
print("-" * 60)
print("Shape:", filtered.shape)
print("Columns:")
print(filtered.columns.tolist())

print("\nFirst 5 rows:")
print(filtered.head())


print("\nRAW ANNOTATIONS")
print("-" * 60)
print("Shape:", raw.shape)
print("Columns:")
print(raw.columns.tolist())

print("\nFirst 5 rows:")
print(raw.head())


print("\nRATER PROFILES")
print("-" * 60)
print("Shape:", profiles.shape)
print("Columns:")
print(profiles.columns.tolist())

print("\nFirst 5 rows:")
print(profiles.head())


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nFiltered annotations:")
print(filtered.isnull().sum())

print("\nRaw annotations:")
print(raw.isnull().sum())

print("\nRater profiles:")
print(profiles.isnull().sum())


# ---------------------------------------------------------
# UNIQUE SONGS / RATERS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET COUNTS")
print("=" * 60)

print(
    "Unique songs in filtered data:",
    filtered["song_id"].nunique()
)

print(
    "Unique raters in filtered data:",
    filtered["rater_id"].nunique()
)

print(
    "Unique songs in raw data:",
    raw["song_id"].nunique()
)

print(
    "Unique raters in raw data:",
    raw["rater_id"].nunique()
)

print(
    "Rater profiles:",
    profiles["rater_id"].nunique()
)


# ---------------------------------------------------------
# NUMBER OF RATINGS PER SONG
# ---------------------------------------------------------

ratings_per_song = (
    filtered
    .groupby("song_id")
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("NUMBER OF RATINGS PER SONG")
print("=" * 60)

print(ratings_per_song)


# ---------------------------------------------------------
# INSPECT VALENCE / AROUSAL ARRAYS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EXAMPLE EMOTION ANNOTATION")
print("=" * 60)

example = filtered.iloc[0]

print("Song:", example["song_id"])
print("Rater:", example["rater_id"])

print("\nValence type:")
print(type(example["valence"]))

print("Number of valence measurements:")
print(len(example["valence"]))

print("First 10 valence measurements:")
print(example["valence"][:10])

print("\nNumber of arousal measurements:")
print(len(example["arousal"]))

print("First 10 arousal measurements:")
print(example["arousal"][:10])


# ---------------------------------------------------------
# MERGE RATINGS WITH LISTENER PROFILES
# ---------------------------------------------------------

merged = filtered.merge(
    profiles,
    on="rater_id",
    how="left"
)

print("\n" + "=" * 60)
print("MERGED DATA")
print("=" * 60)

print("Shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nFirst 5 rows:")
print(merged.head())


# ---------------------------------------------------------
# SAVE MERGED DATA
# ---------------------------------------------------------

merged_path = DATA_DIR / "merp_merged.parquet"

merged.to_parquet(
    merged_path,
    index=False
)

print("\n" + "=" * 60)
print("FINISHED")
print("=" * 60)

print(f"Merged dataset saved to: {merged_path}")