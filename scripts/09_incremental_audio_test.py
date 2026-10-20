import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

from sklearn.metrics import mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data/MERP"

DATA_FILE = os.path.join(
    DATA_FOLDER,
    "modelling_dataset.csv"
)

PREDICTIONS_FILE = os.path.join(
    DATA_FOLDER,
    "corrected_combined_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "incremental_audio_permutation.csv"
)

N_PERMUTATIONS = 200
RANDOM_SEED = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

observed_predictions = pd.read_csv(
    PREDICTIONS_FILE
)

TARGET = "overall_disagreement"

CONTROL_COLUMNS = [
    "mean_valence",
    "mean_arousal",
    "n_raters"
]


# ============================================================
# AUDIO COLUMNS
# ============================================================

non_audio_columns = {
    "song_id",
    "n_raters",
    "mean_valence",
    "mean_arousal",
    "valence_extremity",
    "arousal_extremity",
    "valence_disagreement",
    "arousal_disagreement",
    "overall_disagreement"
}

AUDIO_COLUMNS = [
    c for c in df.columns
    if c not in non_audio_columns
]


print("=" * 70)
print("INCREMENTAL AUDIO PERMUTATION TEST")
print("=" * 70)

print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Controls:", CONTROL_COLUMNS)
print("Permutations:", N_PERMUTATIONS)


# ============================================================
# TARGET
# ============================================================

y = df[TARGET].to_numpy()


# ============================================================
# OBSERVED CONTROL PERFORMANCE
# ============================================================

control_pred = observed_predictions[
    "controls_ridge"
].to_numpy()

combined_pred = observed_predictions[
    "audio_controls_ridge"
].to_numpy()


control_mae = mean_absolute_error(
    y,
    control_pred
)

combined_mae = mean_absolute_error(
    y,
    combined_pred
)


control_rho, _ = spearmanr(
    y,
    control_pred
)

combined_rho, _ = spearmanr(
    y,
    combined_pred
)


# Positive value = audio helped
observed_mae_gain = (
    control_mae
    - combined_mae
)

observed_rho_gain = (
    combined_rho
    - control_rho
)


print("\nObserved results:")

print(
    f"Controls MAE:              "
    f"{control_mae:.6f}"
)

print(
    f"Audio + Controls MAE:      "
    f"{combined_mae:.6f}"
)

print(
    f"MAE gain from audio:        "
    f"{observed_mae_gain:+.6f}"
)

print()

print(
    f"Controls Spearman:          "
    f"{control_rho:.6f}"
)

print(
    f"Audio + Controls Spearman:  "
    f"{combined_rho:.6f}"
)

print(
    f"Spearman gain from audio:   "
    f"{observed_rho_gain:+.6f}"
)


# ============================================================
# CV
# ============================================================

outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)

inner_cv = KFold(
    n_splits=4,
    shuffle=True,
    random_state=RANDOM_SEED
)

outer_splits = list(
    outer_cv.split(df)
)


# ============================================================
# PREPROCESSOR
#
# AUDIO:
# standardise -> PCA
#
# CONTROLS:
# standardise only
# ============================================================

audio_pipeline = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "pca",
        PCA()
    )
])


preprocessor = ColumnTransformer([
    (
        "audio",
        audio_pipeline,
        AUDIO_COLUMNS
    ),

    (
        "controls",
        StandardScaler(),
        CONTROL_COLUMNS
    )
])


combined_model = Pipeline([
    (
        "preprocess",
        preprocessor
    ),

    (
        "model",
        Ridge()
    )
])


parameter_grid = {

    "preprocess__audio__pca__n_components": [
        0.80,
        0.90,
        0.95
    ],

    "model__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


# ============================================================
# FUNCTION:
# NESTED CV WITH GIVEN AUDIO TABLE
# ============================================================

def run_combined_model(
    audio_data
):

    # Controls remain attached to their real songs.
    # Only audio rows are changed.

    X = pd.concat(
        [
            audio_data.reset_index(drop=True),

            df[
                CONTROL_COLUMNS
            ].reset_index(drop=True)
        ],
        axis=1
    )


    predictions = np.zeros(
        len(df)
    )


    for train_idx, test_idx in outer_splits:

        search = GridSearchCV(
            estimator=combined_model,

            param_grid=parameter_grid,

            scoring="neg_mean_absolute_error",

            cv=inner_cv,

            n_jobs=-1
        )


        search.fit(
            X.iloc[train_idx],
            y[train_idx]
        )


        predictions[
            test_idx
        ] = search.predict(
            X.iloc[test_idx]
        )


    mae = mean_absolute_error(
        y,
        predictions
    )


    rho, _ = spearmanr(
        y,
        predictions
    )


    mae_gain = (
        control_mae
        - mae
    )


    rho_gain = (
        rho
        - control_rho
    )


    return (
        mae,
        rho,
        mae_gain,
        rho_gain
    )


# ============================================================
# PERMUTATION
#
# Shuffle complete audio rows between songs.
#
# This preserves relationships between acoustic features
# themselves, but destroys the correct song-audio pairing.
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

audio_original = df[
    AUDIO_COLUMNS
].copy()


results = []


print(
    "\nRunning audio-row permutations..."
)


for permutation in range(
    N_PERMUTATIONS
):

    shuffled_indices = rng.permutation(
        len(df)
    )


    shuffled_audio = (
        audio_original
        .iloc[shuffled_indices]
        .reset_index(drop=True)
    )


    (
        mae,
        rho,
        mae_gain,
        rho_gain
    ) = run_combined_model(
        shuffled_audio
    )


    results.append({

        "permutation":
            permutation + 1,

        "combined_mae":
            mae,

        "combined_spearman":
            rho,

        "mae_gain_over_controls":
            mae_gain,

        "spearman_gain_over_controls":
            rho_gain
    })


    if (
        (permutation + 1) % 20
        == 0
    ):

        print(
            f"Completed "
            f"{permutation + 1}/"
            f"{N_PERMUTATIONS}"
        )


results = pd.DataFrame(
    results
)


# ============================================================
# P VALUES
# ============================================================

mae_p = (
    1
    +
    np.sum(
        results[
            "mae_gain_over_controls"
        ]
        >= observed_mae_gain
    )
) / (
    N_PERMUTATIONS
    + 1
)


rho_p = (
    1
    +
    np.sum(
        results[
            "spearman_gain_over_controls"
        ]
        >= observed_rho_gain
    )
) / (
    N_PERMUTATIONS
    + 1
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INCREMENTAL AUDIO TEST RESULTS")
print("=" * 70)


print("\nOBSERVED")

print(
    f"MAE gain from audio: "
    f"{observed_mae_gain:+.6f}"
)

print(
    f"Spearman gain from audio: "
    f"{observed_rho_gain:+.6f}"
)


print("\nNULL DISTRIBUTION — MAE GAIN")

print(
    results[
        "mae_gain_over_controls"
    ].describe()
)


print(
    "\nPermutation p-value "
    "(audio MAE added value):",
    mae_p
)


print("\nNULL DISTRIBUTION — SPEARMAN GAIN")

print(
    results[
        "spearman_gain_over_controls"
    ].describe()
)


print(
    "\nPermutation p-value "
    "(audio ranking added value):",
    rho_p
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PLOT SPEARMAN ADDED VALUE
#
# This is currently the more interesting metric because the
# observed combined Ridge model slightly improved ranking.
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    results[
        "spearman_gain_over_controls"
    ],
    bins=25,
    edgecolor="black"
)

plt.axvline(
    observed_rho_gain,
    linestyle="--",
    linewidth=2,
    label=(
        "Observed gain = "
        f"{observed_rho_gain:+.3f}"
    )
)

plt.axvline(
    0,
    linewidth=1,
    label="No added value"
)

plt.xlabel(
    "Spearman improvement over controls-only model"
)

plt.ylabel(
    "Number of shuffled-audio datasets"
)

plt.title(
    "Does Correctly Matched Audio Add Information "
    "Beyond Emotion Controls?"
)

plt.legend()

plt.tight_layout()


plot_path = os.path.join(
    DATA_FOLDER,
    "incremental_audio_permutation.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()


print("\nSaved:")
print(OUTPUT_FILE)
print(plot_path)