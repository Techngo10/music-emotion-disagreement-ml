import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

DATA_FOLDER = "data/MERP"

filtered = pd.read_parquet(
    os.path.join(DATA_FOLDER, "annotations_filtered.parquet")
)

raw = pd.read_parquet(
    os.path.join(DATA_FOLDER, "annotations_raw.parquet")
)

profiles = pd.read_parquet(
    os.path.join(DATA_FOLDER, "rater_profiles.parquet")
)


# ---------------------------------------------------------
# 1. BASIC DATA QUALITY CHECKS
# ---------------------------------------------------------

print("=" * 70)
print("DATA QUALITY")
print("=" * 70)

print("\nRaw rows:", len(raw))
print("Filtered rows:", len(filtered))

print("Raw raters:", raw["rater_id"].nunique())
print("Filtered raters:", filtered["rater_id"].nunique())

print("Songs:", filtered["song_id"].nunique())


# ---------------------------------------------------------
# 2. CHECK PROFILE COVERAGE
# ---------------------------------------------------------

filtered_raters = set(filtered["rater_id"].unique())
profile_raters = set(profiles["rater_id"].unique())

matched_profiles = filtered_raters.intersection(profile_raters)
missing_profiles = filtered_raters - profile_raters

print("\n" + "=" * 70)
print("PROFILE COVERAGE")
print("=" * 70)

print("Filtered raters:", len(filtered_raters))
print("Raters with profiles:", len(matched_profiles))
print("Filtered raters without profiles:", len(missing_profiles))


# ---------------------------------------------------------
# 3. IDENTIFY ANCHOR SONGS
# ---------------------------------------------------------

filtered["is_anchor"] = filtered["song_id"].str.startswith("deam_")

anchor_songs = sorted(
    filtered.loc[filtered["is_anchor"], "song_id"].unique()
)

normal_songs = sorted(
    filtered.loc[~filtered["is_anchor"], "song_id"].unique()
)

print("\n" + "=" * 70)
print("SONG TYPES")
print("=" * 70)

print("DEAM anchor songs:")
print(anchor_songs)

print("\nNumber of anchor songs:", len(anchor_songs))
print("Number of normal study songs:", len(normal_songs))


# ---------------------------------------------------------
# 4. CHECK TIME-SERIES LENGTHS
# ---------------------------------------------------------

filtered["valence_length"] = filtered["valence"].apply(len)
filtered["arousal_length"] = filtered["arousal"].apply(len)

print("\n" + "=" * 70)
print("TIME-SERIES LENGTHS")
print("=" * 70)

print("\nValence length summary:")
print(filtered["valence_length"].describe())

print("\nArousal length summary:")
print(filtered["arousal_length"].describe())


# ---------------------------------------------------------
# 5. CALCULATE LISTENER DISAGREEMENT
# ---------------------------------------------------------

def calculate_song_disagreement(group):

    # Arrays should be aligned after MERP preprocessing.
    # We still use the minimum length as a safety check.
    min_valence_length = min(len(x) for x in group["valence"])
    min_arousal_length = min(len(x) for x in group["arousal"])

    valence = np.stack([
        x[:min_valence_length]
        for x in group["valence"]
    ])

    arousal = np.stack([
        x[:min_arousal_length]
        for x in group["arousal"]
    ])

    # -----------------------------------------------------
    # At each moment:
    # Calculate standard deviation ACROSS listeners.
    #
    # High SD = listeners disagree strongly
    # Low SD  = listeners generally agree
    # -----------------------------------------------------

    valence_time_sd = np.std(
        valence,
        axis=0,
        ddof=1
    )

    arousal_time_sd = np.std(
        arousal,
        axis=0,
        ddof=1
    )

    # Average disagreement throughout the whole song
    valence_disagreement = np.mean(valence_time_sd)
    arousal_disagreement = np.mean(arousal_time_sd)

    # Simple combined measure
    overall_disagreement = (
        valence_disagreement +
        arousal_disagreement
    ) / 2

    return pd.Series({

        "n_raters":
            len(group),

        "time_points":
            min(min_valence_length, min_arousal_length),

        "duration_seconds":
            min(min_valence_length, min_arousal_length) / 10,

        "mean_valence":
            np.mean(valence),

        "mean_arousal":
            np.mean(arousal),

        "valence_disagreement":
            valence_disagreement,

        "arousal_disagreement":
            arousal_disagreement,

        "overall_disagreement":
            overall_disagreement
    })


song_summary = (
    filtered
    .groupby("song_id")
    .apply(
        calculate_song_disagreement,
        include_groups=False
    )
    .reset_index()
)

song_summary["is_anchor"] = (
    song_summary["song_id"]
    .str.startswith("deam_")
)


# ---------------------------------------------------------
# 6. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SONG-LEVEL DISAGREEMENT")
print("=" * 70)

print(
    song_summary[
        [
            "song_id",
            "n_raters",
            "valence_disagreement",
            "arousal_disagreement",
            "overall_disagreement",
            "is_anchor"
        ]
    ]
    .sort_values(
        "overall_disagreement",
        ascending=False
    )
    .to_string(index=False)
)


# ---------------------------------------------------------
# 7. REMOVE DEAM ANCHORS FOR MAIN PROJECT DATASET
# ---------------------------------------------------------

project_songs = song_summary[
    ~song_summary["is_anchor"]
].copy()

print("\n" + "=" * 70)
print("FINAL PROJECT SONG DATA")
print("=" * 70)

print("Songs available for project:", len(project_songs))

print("\nOverall disagreement statistics:")
print(
    project_songs["overall_disagreement"].describe()
)


# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

output_path = os.path.join(
    DATA_FOLDER,
    "song_disagreement_summary.csv"
)

project_songs.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)