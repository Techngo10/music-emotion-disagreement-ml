import os
import pandas as pd
from huggingface_hub import hf_hub_download

# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data/MERP"

ANNOTATION_FILE = os.path.join(
    DATA_FOLDER,
    "annotations_filtered.parquet"
)

AUDIO_FOLDER = os.path.join(
    DATA_FOLDER,
    "audio"
)

REPO_ID = "amaai-lab/MERP"

os.makedirs(
    AUDIO_FOLDER,
    exist_ok=True
)


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
            local_dir=DATA_FOLDER
        )

        print("Saved:", path)

    except Exception as e:

        print(
            f"FAILED: {song_id}"
        )

        print(e)


print("\nFinished.")
print("Audio folder:", AUDIO_FOLDER)