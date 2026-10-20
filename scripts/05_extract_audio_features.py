import platform

# Fix for ARM64 Windows running x64 Python.
# openSMILE/audresample otherwise looks for the wrong DLL folder.
_original_machine = platform.machine
platform.machine = lambda: "AMD64"

import opensmile

platform.machine = _original_machine

import numpy as np
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


# ============================================================
# SETTINGS
# ============================================================

AUDIO_FOLDER = DATA_DIR / "audio"
TARGET_FILE = PROCESSED_DIR / "target_diagnostics_songs.csv"
OUTPUT_FILE = PROCESSED_DIR / "audio_features_260.csv"
FINAL_FILE = PROCESSED_DIR / "modelling_dataset.csv"

# IS13 uses frame-level low-level descriptors.
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS13,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

print("=" * 70)
print("OPENSMILE AUDIO FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# LOAD TARGET DATA
# ============================================================

targets = pd.read_csv(TARGET_FILE)

songs = sorted(targets["song_id"].unique())

print("Songs expected:", len(songs))


# ============================================================
# CHECK AUDIO FILES
# ============================================================

missing_audio = []

for song_id in songs:

    path = AUDIO_FOLDER / f"{song_id}.wav"

    if not path.exists():
        missing_audio.append(song_id)

if missing_audio:

    print("\nMissing audio files:")

    for song in missing_audio:
        print(song)

    raise FileNotFoundError(
        "Some required audio files are missing."
    )

print("All required audio files found.")


# ============================================================
# CONVERT 65 LLD FEATURES INTO 260 FEATURES
# ============================================================

def aggregate_500ms(lld_df):
    """
    IS13 gives low-level descriptors at roughly 10 ms intervals.

    MERP's feature representation uses four summary
    statistics within 500 ms windows.

    For each LLD:
        mean
        std
        min
        max

    65 LLDs x 4 statistics = 260 features.
    """

    # Flatten index and keep only numeric feature columns
    frame_data = lld_df.reset_index(drop=True)

    frame_data = frame_data.select_dtypes(
        include=[np.number]
    )

    n_features = frame_data.shape[1]

    print(
        f"    Low-level descriptors: {n_features}"
    )

    # Approx. 50 x 10 ms frames = 500 ms
    WINDOW_FRAMES = 50

    segment_rows = []

    for start in range(
        0,
        len(frame_data),
        WINDOW_FRAMES
    ):

        window = frame_data.iloc[
            start:start + WINDOW_FRAMES
        ]

        # Ignore tiny incomplete tail windows
        if len(window) < 10:
            continue

        features = {}

        for column in frame_data.columns:

            values = (
                window[column]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(values) == 0:

                mean_value = np.nan
                std_value = np.nan
                min_value = np.nan
                max_value = np.nan

            else:

                mean_value = values.mean()
                std_value = values.std(ddof=0)
                min_value = values.min()
                max_value = values.max()

            name = str(column)

            features[f"{name}_mean"] = mean_value
            features[f"{name}_std"] = std_value
            features[f"{name}_min"] = min_value
            features[f"{name}_max"] = max_value

        segment_rows.append(features)

    return pd.DataFrame(segment_rows)


# ============================================================
# EXTRACT SONG FEATURES
# ============================================================

song_feature_rows = []

for number, song_id in enumerate(
    songs,
    start=1
):

    audio_path = AUDIO_FOLDER / f"{song_id}.wav"

    print(
        f"\n[{number}/{len(songs)}] {song_id}"
    )

    # --------------------------------------------------------
    # FRAME-LEVEL IS13 FEATURES
    # --------------------------------------------------------

    lld = smile.process_file(
        audio_path
    )

    print(
        "    LLD frames:",
        len(lld)
    )

    # --------------------------------------------------------
    # 500 ms FEATURE WINDOWS
    # --------------------------------------------------------

    segments = aggregate_500ms(
        lld
    )

    print(
        "    500 ms segments:",
        len(segments)
    )

    print(
        "    Segment features:",
        segments.shape[1]
    )

    # --------------------------------------------------------
    # SONG-LEVEL REPRESENTATION
    #
    # We currently have many 500 ms segments for a song.
    # Average each acoustic descriptor across the song.
    #
    # Result:
    # one row per song.
    # --------------------------------------------------------

    song_features = (
        segments.mean(axis=0)
    )

    row = {
        "song_id": song_id
    }

    row.update(
        song_features.to_dict()
    )

    song_feature_rows.append(row)


# ============================================================
# CREATE FEATURE TABLE
# ============================================================

audio_features = pd.DataFrame(
    song_feature_rows
)

print("\n" + "=" * 70)
print("AUDIO FEATURE TABLE")
print("=" * 70)

print(
    "Shape:",
    audio_features.shape
)

print(
    "Songs:",
    audio_features["song_id"].nunique()
)

print(
    "Acoustic features:",
    audio_features.shape[1] - 1
)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

feature_columns = [
    c for c in audio_features.columns
    if c != "song_id"
]

audio_features[feature_columns] = (
    audio_features[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

missing_counts = (
    audio_features[feature_columns]
    .isna()
    .sum()
)

problem_columns = missing_counts[
    missing_counts > 0
]

print("\nFeatures containing missing values:")

if len(problem_columns) == 0:

    print("None")

else:

    print(problem_columns)


# ============================================================
# REMOVE FEATURES THAT ARE COMPLETELY EMPTY
# ============================================================

all_nan_columns = [
    column
    for column in feature_columns
    if audio_features[column].isna().all()
]

if all_nan_columns:

    print(
        "\nDropping completely empty features:",
        len(all_nan_columns)
    )

    audio_features = audio_features.drop(
        columns=all_nan_columns
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

feature_columns = [
    c for c in audio_features.columns
    if c != "song_id"
]

constant_features = [
    column
    for column in feature_columns
    if audio_features[column].nunique(
        dropna=True
    ) <= 1
]

print(
    "\nConstant features:",
    len(constant_features)
)

if constant_features:

    audio_features = audio_features.drop(
        columns=constant_features
    )


# ============================================================
# SAVE AUDIO FEATURES
# ============================================================

audio_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved acoustic features:"
)

print(OUTPUT_FILE)


# ============================================================
# MERGE FEATURES WITH TARGET + CONTROLS
# ============================================================

model_data = targets.merge(
    audio_features,
    on="song_id",
    how="inner"
)

print("\n" + "=" * 70)
print("FINAL MODELLING DATASET")
print("=" * 70)

print(
    "Shape:",
    model_data.shape
)

print(
    "Songs:",
    model_data["song_id"].nunique()
)

print(
    "Target:",
    "overall_disagreement"
)

print(
    "Control variables:",
    [
        "mean_valence",
        "mean_arousal",
        "n_raters"
    ]
)

print(
    "Total columns:",
    len(model_data.columns)
)


# ============================================================
# CHECK SONG MATCHING
# ============================================================

if len(model_data) != len(targets):

    print(
        "\nWARNING:"
        " Not every target song matched an audio row."
    )

else:

    print(
        "\nAll target songs successfully matched."
    )


# ============================================================
# SAVE FINAL TABLE
# ============================================================

model_data.to_csv(
    FINAL_FILE,
    index=False
)

print("\nSaved modelling dataset:")
print(FINAL_FILE)

print("\nDONE.")