import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data/MERP"

INPUT_FILE = os.path.join(
    DATA_FOLDER,
    "annotations_filtered.parquet"
)

OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "adjusted_disagreement.csv"
)

N_REPEATS = 1000
RANDOM_SEED = 42

EPSILON = 1e-8


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

# Remove four DEAM anchor tracks
df = df[
    ~df["song_id"].str.startswith("deam_")
].copy()

print("=" * 70)
print("BOUNDED-SCALE ADJUSTED DISAGREEMENT")
print("=" * 70)

print("Songs:", df["song_id"].nunique())


# ============================================================
# CALCULATE SONG STATISTICS
# ============================================================

def calculate_statistics(group):

    common_length = min(
        min(len(x) for x in group["valence"]),
        min(len(x) for x in group["arousal"])
    )

    valence = np.stack([
        np.asarray(
            x[:common_length],
            dtype=float
        )
        for x in group["valence"]
    ])

    arousal = np.stack([
        np.asarray(
            x[:common_length],
            dtype=float
        )
        for x in group["arousal"]
    ])


    # ========================================================
    # MEANS ACROSS LISTENERS
    # ========================================================

    valence_mean = np.mean(
        valence,
        axis=0
    )

    arousal_mean = np.mean(
        arousal,
        axis=0
    )


    # ========================================================
    # POPULATION STANDARD DEVIATION
    #
    # ddof=0 because the theoretical variance bound applies
    # to the distribution variance.
    # ========================================================

    valence_sd = np.std(
        valence,
        axis=0,
        ddof=0
    )

    arousal_sd = np.std(
        arousal,
        axis=0,
        ddof=0
    )


    # ========================================================
    # MAXIMUM POSSIBLE SD GIVEN THE MEAN
    #
    # Ratings are bounded to [-1, +1].
    #
    # Var(X) <= (1 - mean)(mean + 1)
    #        = 1 - mean^2
    # ========================================================

    valence_max_sd = np.sqrt(
        np.maximum(
            1 - valence_mean ** 2,
            0
        )
    )

    arousal_max_sd = np.sqrt(
        np.maximum(
            1 - arousal_mean ** 2,
            0
        )
    )


    # ========================================================
    # NORMALISED DISAGREEMENT
    # ========================================================

    valence_normalised = np.divide(
        valence_sd,
        valence_max_sd,

        out=np.zeros_like(
            valence_sd
        ),

        where=(
            valence_max_sd > EPSILON
        )
    )

    arousal_normalised = np.divide(
        arousal_sd,
        arousal_max_sd,

        out=np.zeros_like(
            arousal_sd
        ),

        where=(
            arousal_max_sd > EPSILON
        )
    )


    # Numerical safety
    valence_normalised = np.clip(
        valence_normalised,
        0,
        1
    )

    arousal_normalised = np.clip(
        arousal_normalised,
        0,
        1
    )


    # ========================================================
    # SONG-LEVEL VALUES
    # ========================================================

    normalised_valence_disagreement = (
        np.mean(valence_normalised)
    )

    normalised_arousal_disagreement = (
        np.mean(arousal_normalised)
    )

    normalised_overall_disagreement = (
        normalised_valence_disagreement
        +
        normalised_arousal_disagreement
    ) / 2


    # Original disagreement for comparison
    original_valence = np.mean(
        valence_sd
    )

    original_arousal = np.mean(
        arousal_sd
    )

    original_overall = (
        original_valence
        +
        original_arousal
    ) / 2


    return {
        "n_raters":
            len(group),

        "mean_valence":
            np.mean(valence_mean),

        "mean_arousal":
            np.mean(arousal_mean),

        "original_disagreement":
            original_overall,

        "normalised_valence_disagreement":
            normalised_valence_disagreement,

        "normalised_arousal_disagreement":
            normalised_arousal_disagreement,

        "normalised_overall_disagreement":
            normalised_overall_disagreement
    }


# ============================================================
# FULL DATA TARGET
# ============================================================

rows = []

for song_id, group in df.groupby(
    "song_id"
):

    stats = calculate_statistics(
        group
    )

    stats["song_id"] = song_id

    rows.append(stats)


songs = pd.DataFrame(rows)


print("\n" + "=" * 70)
print("NORMALISED TARGET")
print("=" * 70)

print(
    songs[
        "normalised_overall_disagreement"
    ].describe()
)


# ============================================================
# CONFOUND CHECK
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH CONTROLS")
print("=" * 70)


for variable in [
    "mean_valence",
    "mean_arousal",
    "n_raters"
]:

    pearson_r, pearson_p = pearsonr(
        songs[variable],
        songs[
            "normalised_overall_disagreement"
        ]
    )

    spearman_r, spearman_p = spearmanr(
        songs[variable],
        songs[
            "normalised_overall_disagreement"
        ]
    )

    print(
        f"{variable:15s} "
        f"Pearson={pearson_r: .3f} "
        f"Spearman={spearman_r: .3f}"
    )


# ============================================================
# ORIGINAL VS NORMALISED
# ============================================================

r, p = spearmanr(
    songs["original_disagreement"],
    songs[
        "normalised_overall_disagreement"
    ]
)

print("\nOriginal vs normalised target")
print("Spearman:", r)


# ============================================================
# SPLIT-HALF RELIABILITY
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

song_ids = sorted(
    df["song_id"].unique()
)

results = []


print("\nRunning split-half reliability...")


for repetition in range(
    N_REPEATS
):

    half1 = []
    half2 = []


    for song_id in song_ids:

        song = df[
            df["song_id"] == song_id
        ]

        indices = rng.permutation(
            len(song)
        )

        split = len(indices) // 2

        group1 = song.iloc[
            indices[:split]
        ]

        group2 = song.iloc[
            indices[split:]
        ]

        stat1 = calculate_statistics(
            group1
        )

        stat2 = calculate_statistics(
            group2
        )

        half1.append(
            stat1[
                "normalised_overall_disagreement"
            ]
        )

        half2.append(
            stat2[
                "normalised_overall_disagreement"
            ]
        )


    rho, _ = spearmanr(
        half1,
        half2
    )

    r, _ = pearsonr(
        half1,
        half2
    )

    results.append({
        "spearman": rho,
        "pearson": r
    })


results = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

median_spearman = (
    results["spearman"].median()
)

median_pearson = (
    results["pearson"].median()
)

spearman_brown = (
    2 * median_pearson
) / (
    1 + median_pearson
)


print("\n" + "=" * 70)
print("NORMALISED TARGET RELIABILITY")
print("=" * 70)

print(
    "Mean Spearman:",
    results["spearman"].mean()
)

print(
    "Median Spearman:",
    median_spearman
)

print(
    "Central 95%:",
    (
        results[
            "spearman"
        ].quantile(0.025),

        results[
            "spearman"
        ].quantile(0.975)
    )
)

print(
    "Median Pearson:",
    median_pearson
)

print(
    "Spearman-Brown:",
    spearman_brown
)


# ============================================================
# SAVE
# ============================================================

songs.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)