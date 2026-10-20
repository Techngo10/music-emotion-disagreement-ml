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

import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = DATA_DIR / "annotations_filtered.parquet"

N_REPEATS = 1000
TOP_K = 10
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

print("=" * 70)
print("SPLIT-HALF RELIABILITY ANALYSIS")
print("=" * 70)

print("\nRows loaded:", len(df))
print("Unique songs:", df["song_id"].nunique())
print("Unique raters:", df["rater_id"].nunique())


# ============================================================
# REMOVE THE FOUR DEAM ANCHOR SONGS
# ============================================================

df = df[
    ~df["song_id"].str.startswith("deam_")
].copy()

print("\nAfter removing DEAM anchor songs:")
print("Songs:", df["song_id"].nunique())
print("Rows:", len(df))


# ============================================================
# DISAGREEMENT FUNCTION
# ============================================================

def calculate_disagreement(group):
    """
    Calculate the same disagreement score used previously.

    For every time point:
        standard deviation across listeners

    Then:
        average across the song

    Final score:
        average of valence and arousal disagreement
    """

    if len(group) < 2:
        return np.nan

    # People rating the same song should normally have
    # aligned arrays, but use minimum length as a safeguard.
    min_valence_length = min(
        len(x) for x in group["valence"]
    )

    min_arousal_length = min(
        len(x) for x in group["arousal"]
    )

    common_length = min(
        min_valence_length,
        min_arousal_length
    )

    valence = np.stack([
        np.asarray(x[:common_length], dtype=float)
        for x in group["valence"]
    ])

    arousal = np.stack([
        np.asarray(x[:common_length], dtype=float)
        for x in group["arousal"]
    ])

    # Standard deviation BETWEEN listeners
    # at each point in the song
    valence_sd = np.std(
        valence,
        axis=0,
        ddof=1
    )

    arousal_sd = np.std(
        arousal,
        axis=0,
        ddof=1
    )

    valence_disagreement = np.mean(valence_sd)
    arousal_disagreement = np.mean(arousal_sd)

    overall_disagreement = (
        valence_disagreement
        + arousal_disagreement
    ) / 2

    return overall_disagreement


# ============================================================
# FULL-DATA DISAGREEMENT
# ============================================================

full_scores = (
    df
    .groupby("song_id")
    .apply(
        calculate_disagreement,
        include_groups=False
    )
    .rename("full_disagreement")
    .reset_index()
)

print("\nFull-data disagreement:")
print(full_scores["full_disagreement"].describe())


# ============================================================
# REPEATED SPLIT-HALF TEST
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

results = []
split_scores = []

songs = sorted(df["song_id"].unique())

print(
    f"\nRunning {N_REPEATS} random split-half repetitions..."
)


for repetition in range(N_REPEATS):

    half1_scores = {}
    half2_scores = {}

    for song_id in songs:

        song_data = df[
            df["song_id"] == song_id
        ]

        n = len(song_data)

        # Randomly shuffle the rater rows
        indices = rng.permutation(n)

        split_point = n // 2

        half1_indices = indices[:split_point]
        half2_indices = indices[split_point:]

        half1 = song_data.iloc[half1_indices]
        half2 = song_data.iloc[half2_indices]

        half1_scores[song_id] = (
            calculate_disagreement(half1)
        )

        half2_scores[song_id] = (
            calculate_disagreement(half2)
        )

    # Put songs in identical ordering
    scores1 = np.array([
        half1_scores[s]
        for s in songs
    ])

    scores2 = np.array([
        half2_scores[s]
        for s in songs
    ])

    # --------------------------------------------------------
    # SPEARMAN:
    # Do both groups RANK the songs similarly?
    # --------------------------------------------------------

    spearman_rho, _ = spearmanr(
        scores1,
        scores2
    )

    # --------------------------------------------------------
    # PEARSON:
    # Are the actual disagreement values linearly similar?
    # --------------------------------------------------------

    pearson_r, _ = pearsonr(
        scores1,
        scores2
    )

    # --------------------------------------------------------
    # TOP-K OVERLAP
    #
    # Are the most disputed songs the same?
    # --------------------------------------------------------

    ranking1 = np.argsort(scores1)[::-1]
    ranking2 = np.argsort(scores2)[::-1]

    top1 = set(ranking1[:TOP_K])
    top2 = set(ranking2[:TOP_K])

    overlap = len(
        top1.intersection(top2)
    )

    results.append({
        "repetition": repetition,
        "spearman": spearman_rho,
        "pearson": pearson_r,
        "top10_overlap": overlap
    })

    # Save individual song scores so we can plot
    # a representative split later.
    for i, song_id in enumerate(songs):
        split_scores.append({
            "repetition": repetition,
            "song_id": song_id,
            "half1": scores1[i],
            "half2": scores2[i]
        })


# ============================================================
# RESULTS TABLES
# ============================================================

results_df = pd.DataFrame(results)

split_scores_df = pd.DataFrame(
    split_scores
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT-HALF RESULTS")
print("=" * 70)

print("\nSPEARMAN RANK CORRELATION")

print(
    "Mean:",
    results_df["spearman"].mean()
)

print(
    "Median:",
    results_df["spearman"].median()
)

print(
    "2.5th percentile:",
    results_df["spearman"].quantile(0.025)
)

print(
    "97.5th percentile:",
    results_df["spearman"].quantile(0.975)
)


print("\nPEARSON CORRELATION")

print(
    "Mean:",
    results_df["pearson"].mean()
)

print(
    "Median:",
    results_df["pearson"].median()
)

print(
    "2.5th percentile:",
    results_df["pearson"].quantile(0.025)
)

print(
    "97.5th percentile:",
    results_df["pearson"].quantile(0.975)
)


print(f"\nTOP-{TOP_K} SONG OVERLAP")

print(
    "Mean number appearing in both halves:",
    results_df["top10_overlap"].mean()
)

print(
    "Median:",
    results_df["top10_overlap"].median()
)

print(
    "Minimum:",
    results_df["top10_overlap"].min()
)

print(
    "Maximum:",
    results_df["top10_overlap"].max()
)

# Expected overlap if rankings were random:
expected_random_overlap = (
    TOP_K * TOP_K
) / len(songs)

print(
    "Expected overlap by chance:",
    expected_random_overlap
)


# ============================================================
# SPEARMAN-BROWN CORRECTION
# ============================================================

median_pearson = (
    results_df["pearson"].median()
)

spearman_brown = (
    2 * median_pearson
) / (
    1 + median_pearson
)

print("\n" + "=" * 70)
print("SPEARMAN-BROWN RELIABILITY")
print("=" * 70)

print(
    "Using median split-half Pearson correlation:",
    median_pearson
)

print(
    "Spearman-Brown corrected reliability:",
    spearman_brown
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = TABLES_DIR / "split_half_reliability.csv"

results_df.to_csv(
    results_path,
    index=False
)

split_scores_path = TABLES_DIR / "split_half_song_scores.csv"

split_scores_df.to_csv(
    split_scores_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(split_scores_path)


# ============================================================
# GRAPH 1
# DISTRIBUTION OF SPEARMAN CORRELATIONS
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    results_df["spearman"],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    results_df["spearman"].median(),
    linestyle="--",
    label=(
        "Median = "
        f"{results_df['spearman'].median():.3f}"
    )
)

plt.xlabel("Spearman correlation between listener halves")
plt.ylabel("Number of random splits")

plt.title(
    "Split-Half Reliability of Emotional Disagreement"
)

plt.legend()
plt.tight_layout()

plot1_path = FIGURES_DIR / "split_half_spearman_distribution.png"

plt.savefig(
    plot1_path,
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2
# REPRESENTATIVE SPLIT
# ============================================================

median_spearman = (
    results_df["spearman"].median()
)

representative_index = (
    results_df["spearman"]
    .sub(median_spearman)
    .abs()
    .idxmin()
)

representative_rep = int(
    results_df.loc[
        representative_index,
        "repetition"
    ]
)

representative = split_scores_df[
    split_scores_df["repetition"]
    == representative_rep
]

representative_rho = results_df.loc[
    representative_index,
    "spearman"
]

plt.figure(figsize=(7, 6))

plt.scatter(
    representative["half1"],
    representative["half2"]
)

plt.xlabel(
    "Disagreement score — Listener Half 1"
)

plt.ylabel(
    "Disagreement score — Listener Half 2"
)

plt.title(
    "Representative Split-Half Comparison\n"
    f"Spearman ρ = {representative_rho:.3f}"
)

plt.tight_layout()

plot2_path = FIGURES_DIR / "split_half_representative_scatter.png"

plt.savefig(
    plot2_path,
    dpi=300
)

plt.show()


print("\nPlots saved:")
print(plot1_path)
print(plot2_path)