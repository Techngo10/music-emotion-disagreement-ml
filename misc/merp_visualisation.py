import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_FOLDER = "data/MERP"

songs = pd.read_csv(
    os.path.join(DATA_FOLDER, "song_disagreement_summary.csv")
)

# ---------------------------------------------------------
# GRAPH 1
# Distribution of disagreement scores
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    songs["overall_disagreement"],
    bins=10,
    edgecolor="black"
)

plt.xlabel("Overall Listener Disagreement")
plt.ylabel("Number of Songs")
plt.title("Distribution of Emotional Disagreement Across Songs")

plt.tight_layout()

plt.savefig(
    os.path.join(DATA_FOLDER, "disagreement_distribution.png"),
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# GRAPH 2
# Valence vs arousal disagreement
# ---------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    songs["valence_disagreement"],
    songs["arousal_disagreement"]
)

plt.xlabel("Valence Disagreement")
plt.ylabel("Arousal Disagreement")
plt.title("Valence vs Arousal Disagreement")

plt.tight_layout()

plt.savefig(
    os.path.join(DATA_FOLDER, "valence_vs_arousal.png"),
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# GRAPH 3
# Number of raters vs disagreement
# ---------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    songs["n_raters"],
    songs["overall_disagreement"]
)

plt.xlabel("Number of Raters")
plt.ylabel("Overall Listener Disagreement")
plt.title("Number of Raters vs Emotional Disagreement")

plt.tight_layout()

plt.savefig(
    os.path.join(DATA_FOLDER, "raters_vs_disagreement.png"),
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# GRAPH 4
# Highest and lowest disagreement songs
# ---------------------------------------------------------

sorted_songs = songs.sort_values(
    "overall_disagreement",
    ascending=False
)

selected = pd.concat([
    sorted_songs.head(3),
    sorted_songs.tail(3)
])

selected = selected.sort_values(
    "overall_disagreement"
)

plt.figure(figsize=(5, 3))

plt.barh(
    selected["song_id"],
    selected["overall_disagreement"]
)

plt.xlabel("Overall Listener Disagreement")
plt.ylabel("Song")
plt.title("Songs with Highest and Lowest Emotional Disagreement")

plt.tight_layout()

plt.savefig(
    os.path.join(DATA_FOLDER, "highest_lowest_disagreement.png"),
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# SIMPLE CORRELATIONS
# ---------------------------------------------------------

print("=" * 60)
print("CORRELATIONS")
print("=" * 60)

print(
    "\nValence vs Arousal disagreement:",
    songs["valence_disagreement"].corr(
        songs["arousal_disagreement"]
    )
)

print(
    "Number of raters vs disagreement:",
    songs["n_raters"].corr(
        songs["overall_disagreement"]
    )
)