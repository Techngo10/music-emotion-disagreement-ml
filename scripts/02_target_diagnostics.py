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


from scipy.stats import spearmanr, pearsonr


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = DATA_DIR / "annotations_filtered.parquet"

N_REPEATS = 1000
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

# Remove four DEAM quality-control anchor songs
df = df[
    ~df["song_id"].str.startswith("deam_")
].copy()

print("=" * 70)
print("TARGET DIAGNOSTICS")
print("=" * 70)

print("Songs:", df["song_id"].nunique())
print("Annotation rows:", len(df))


# ============================================================
# FUNCTION FOR SONG STATISTICS
# ============================================================

def calculate_statistics(group):

    if len(group) < 2:
        return None

    # Make sure all listener arrays are aligned
    common_length = min(
        min(len(x) for x in group["valence"]),
        min(len(x) for x in group["arousal"])
    )

    valence = np.stack([
        np.asarray(x[:common_length], dtype=float)
        for x in group["valence"]
    ])

    arousal = np.stack([
        np.asarray(x[:common_length], dtype=float)
        for x in group["arousal"]
    ])

    # --------------------------------------------------------
    # MEAN EMOTION AT EACH TIME POINT
    # --------------------------------------------------------

    mean_valence_time = np.mean(
        valence,
        axis=0
    )

    mean_arousal_time = np.mean(
        arousal,
        axis=0
    )


    # --------------------------------------------------------
    # DISAGREEMENT AT EACH TIME POINT
    # --------------------------------------------------------

    valence_sd_time = np.std(
        valence,
        axis=0,
        ddof=1
    )

    arousal_sd_time = np.std(
        arousal,
        axis=0,
        ddof=1
    )


    # --------------------------------------------------------
    # SONG-LEVEL VALUES
    # --------------------------------------------------------

    mean_valence = np.mean(
        mean_valence_time
    )

    mean_arousal = np.mean(
        mean_arousal_time
    )

    # How close the emotion tends to be to the boundaries
    # of the -1 to +1 rating scale
    valence_extremity = np.mean(
        np.abs(mean_valence_time)
    )

    arousal_extremity = np.mean(
        np.abs(mean_arousal_time)
    )

    valence_disagreement = np.mean(
        valence_sd_time
    )

    arousal_disagreement = np.mean(
        arousal_sd_time
    )

    overall_disagreement = (
        valence_disagreement
        + arousal_disagreement
    ) / 2

    return {
        "n_raters": len(group),

        "mean_valence": mean_valence,
        "mean_arousal": mean_arousal,

        "valence_extremity": valence_extremity,
        "arousal_extremity": arousal_extremity,

        "valence_disagreement":
            valence_disagreement,

        "arousal_disagreement":
            arousal_disagreement,

        "overall_disagreement":
            overall_disagreement
    }


# ============================================================
# FULL SONG-LEVEL TABLE
# ============================================================

rows = []

for song_id, group in df.groupby("song_id"):

    stats = calculate_statistics(group)

    stats["song_id"] = song_id

    rows.append(stats)


song_data = pd.DataFrame(rows)

columns = [
    "song_id",
    "n_raters",
    "mean_valence",
    "mean_arousal",
    "valence_extremity",
    "arousal_extremity",
    "valence_disagreement",
    "arousal_disagreement",
    "overall_disagreement"
]

song_data = song_data[columns]


print("\n" + "=" * 70)
print("SONG DATA")
print("=" * 70)

print(song_data.head())

print("\nDisagreement summary:")

print(
    song_data[
        [
            "valence_disagreement",
            "arousal_disagreement",
            "overall_disagreement"
        ]
    ].describe()
)


# ============================================================
# CONFOUND CORRELATIONS
# ============================================================

potential_confounds = [
    "n_raters",
    "mean_valence",
    "mean_arousal",
    "valence_extremity",
    "arousal_extremity"
]

targets = [
    "valence_disagreement",
    "arousal_disagreement",
    "overall_disagreement"
]


print("\n" + "=" * 70)
print("CORRELATIONS WITH POSSIBLE CONFOUNDS")
print("=" * 70)


correlation_rows = []

for target in targets:

    print(f"\nTARGET: {target}")

    for variable in potential_confounds:

        pearson_r, pearson_p = pearsonr(
            song_data[variable],
            song_data[target]
        )

        spearman_r, spearman_p = spearmanr(
            song_data[variable],
            song_data[target]
        )

        print(
            f"{variable:22s} "
            f"Pearson={pearson_r: .3f} "
            f"Spearman={spearman_r: .3f}"
        )

        correlation_rows.append({
            "target": target,
            "variable": variable,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p
        })


correlations = pd.DataFrame(
    correlation_rows
)


# ============================================================
# SPLIT-HALF RELIABILITY FOR EACH TARGET
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

songs = sorted(
    df["song_id"].unique()
)

reliability_results = []


print("\n" + "=" * 70)
print("RUNNING SPLIT-HALF TESTS")
print("=" * 70)


for repetition in range(N_REPEATS):

    half1 = {
        "valence": [],
        "arousal": [],
        "overall": []
    }

    half2 = {
        "valence": [],
        "arousal": [],
        "overall": []
    }


    for song_id in songs:

        song = df[
            df["song_id"] == song_id
        ]

        indices = rng.permutation(
            len(song)
        )

        split_point = (
            len(indices) // 2
        )

        group1 = song.iloc[
            indices[:split_point]
        ]

        group2 = song.iloc[
            indices[split_point:]
        ]

        s1 = calculate_statistics(
            group1
        )

        s2 = calculate_statistics(
            group2
        )

        half1["valence"].append(
            s1["valence_disagreement"]
        )

        half2["valence"].append(
            s2["valence_disagreement"]
        )

        half1["arousal"].append(
            s1["arousal_disagreement"]
        )

        half2["arousal"].append(
            s2["arousal_disagreement"]
        )

        half1["overall"].append(
            s1["overall_disagreement"]
        )

        half2["overall"].append(
            s2["overall_disagreement"]
        )


    for target in [
        "valence",
        "arousal",
        "overall"
    ]:

        x = np.asarray(
            half1[target]
        )

        y = np.asarray(
            half2[target]
        )

        spearman_rho, _ = spearmanr(
            x,
            y
        )

        pearson_r, _ = pearsonr(
            x,
            y
        )

        reliability_results.append({
            "repetition": repetition,
            "target": target,
            "spearman": spearman_rho,
            "pearson": pearson_r
        })


# ============================================================
# SUMMARISE RELIABILITY
# ============================================================

reliability = pd.DataFrame(
    reliability_results
)


print("\n" + "=" * 70)
print("RELIABILITY BY TARGET")
print("=" * 70)


summary_rows = []

for target in [
    "valence",
    "arousal",
    "overall"
]:

    subset = reliability[
        reliability["target"] == target
    ]

    median_spearman = (
        subset["spearman"].median()
    )

    median_pearson = (
        subset["pearson"].median()
    )

    spearman_brown = (
        2 * median_pearson
    ) / (
        1 + median_pearson
    )

    print(f"\n{target.upper()}")

    print(
        "Mean Spearman:",
        subset["spearman"].mean()
    )

    print(
        "Median Spearman:",
        median_spearman
    )

    print(
        "Spearman central 95%:",
        (
            subset["spearman"].quantile(0.025),
            subset["spearman"].quantile(0.975)
        )
    )

    print(
        "Median Pearson:",
        median_pearson
    )

    print(
        "Spearman-Brown reliability:",
        spearman_brown
    )

    summary_rows.append({
        "target": target,
        "mean_spearman":
            subset["spearman"].mean(),

        "median_spearman":
            median_spearman,

        "lower_2_5":
            subset["spearman"].quantile(0.025),

        "upper_97_5":
            subset["spearman"].quantile(0.975),

        "median_pearson":
            median_pearson,

        "spearman_brown":
            spearman_brown
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

song_path = PROCESSED_DIR / "target_diagnostics_songs.csv"
correlation_path = TABLES_DIR / "target_confound_correlations.csv"
reliability_path = TABLES_DIR / "target_reliability_comparison.csv"


song_data.to_csv(
    song_path,
    index=False
)

correlations.to_csv(
    correlation_path,
    index=False
)

summary.to_csv(
    reliability_path,
    index=False
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(song_path)
print(correlation_path)
print(reliability_path)