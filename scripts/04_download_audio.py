import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MERP"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import hf_hub_download

# ============================================================
# SETTINGS
# ============================================================

ANNOTATION_FILE = DATA_DIR / "annotations_filtered.parquet"
AUDIO_FOLDER = DATA_DIR / "audio"
REPO_ID = "amaai-lab/MERP"

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET THE 50 STUDY SONG IDS
# ============================================================

df = pd.read_parquet(
    ANNOTATION_FILE
)

songs = sorted(
    df.loc[
        ~df["song_id"].str.startswith("deam_"),
        "song_id"
    ].unique()
)

print("Number of study songs:", len(songs))


# ============================================================
# DOWNLOAD AUDIO
# ============================================================

for i, song_id in enumerate(songs, start=1):

    filename = f"audio/{song_id}.wav"

    print(
        f"[{i}/{len(songs)}] Downloading {song_id}"
    )

    try:

        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=DATA_DIR
        )

        print("Saved:", path)

    except Exception as e:

        print(
            f"FAILED: {song_id}"
        )

        print(e)


print("\nFinished.")
print("Audio folder:", AUDIO_FOLDER)