import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data/MERP"

INPUT_FILE = os.path.join(
    DATA_FOLDER,
    "modelling_dataset.csv"
)

PREDICTIONS_FILE = os.path.join(
    DATA_FOLDER,
    "nested_cv_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "audio_permutation_results.csv"
)

RANDOM_SEED = 42

# Start with 200.
# This gives a minimum attainable p-value of about 0.005.
N_PERMUTATIONS = 200


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

previous_predictions = pd.read_csv(
    PREDICTIONS_FILE
)

TARGET = "overall_disagreement"


# ============================================================
# FIND AUDIO COLUMNS
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
    column
    for column in df.columns
    if column not in non_audio_columns
]


X_audio = df[
    AUDIO_COLUMNS
].to_numpy()

y = df[TARGET].to_numpy()


print("=" * 70)
print("AUDIO PERMUTATION TEST")
print("=" * 70)

print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Permutations:", N_PERMUTATIONS)


# ============================================================
# OBSERVED PERFORMANCE
#
# Use the predictions from our original nested-CV analysis.
# ============================================================

observed_baseline_mae = mean_absolute_error(
    previous_predictions["actual"],
    previous_predictions["mean_baseline"]
)

observed_audio_mae = mean_absolute_error(
    previous_predictions["actual"],
    previous_predictions["audio_ridge"]
)

observed_improvement = (
    observed_baseline_mae
    - observed_audio_mae
)


observed_spearman, _ = spearmanr(
    previous_predictions["actual"],
    previous_predictions["audio_ridge"]
)


print("\nObserved results:")

print(
    "Mean baseline MAE:",
    observed_baseline_mae
)

print(
    "Audio Ridge MAE:",
    observed_audio_mae
)

print(
    "MAE improvement:",
    observed_improvement
)

print(
    "Audio Ridge Spearman:",
    observed_spearman
)


# ============================================================
# MODEL
# ============================================================

pipeline = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "pca",
        PCA()
    ),
    (
        "model",
        Ridge()
    )
])


parameter_grid = {

    "pca__n_components": [
        0.80,
        0.90,
        0.95
    ],

    "model__alpha": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


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


# Store outer splits so every permutation uses
# exactly the same folds.
outer_splits = list(
    outer_cv.split(X_audio)
)


# ============================================================
# FUNCTION:
# RUN NESTED CV FOR ONE TARGET VECTOR
# ============================================================

def run_nested_cv(target):

    model_predictions = np.zeros(
        len(target)
    )

    baseline_predictions = np.zeros(
        len(target)
    )


    for train_idx, test_idx in outer_splits:

        y_train = target[train_idx]


        # ----------------------------------------------------
        # MEAN BASELINE
        # ----------------------------------------------------

        baseline_predictions[
            test_idx
        ] = np.mean(y_train)


        # ----------------------------------------------------
        # AUDIO RIDGE
        # ----------------------------------------------------

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameter_grid,

            scoring="neg_mean_absolute_error",

            cv=inner_cv,

            n_jobs=-1
        )


        search.fit(
            X_audio[train_idx],
            y_train
        )


        model_predictions[
            test_idx
        ] = search.predict(
            X_audio[test_idx]
        )


    baseline_mae = mean_absolute_error(
        target,
        baseline_predictions
    )

    model_mae = mean_absolute_error(
        target,
        model_predictions
    )


    improvement = (
        baseline_mae
        - model_mae
    )


    rho, _ = spearmanr(
        target,
        model_predictions
    )


    return (
        baseline_mae,
        model_mae,
        improvement,
        rho
    )


# ============================================================
# PERMUTATIONS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

results = []


print(
    "\nRunning permutations..."
)


for permutation in range(
    N_PERMUTATIONS
):

    shuffled_y = rng.permutation(
        y
    )


    (
        baseline_mae,
        model_mae,
        improvement,
        rho
    ) = run_nested_cv(
        shuffled_y
    )


    results.append({

        "permutation":
            permutation + 1,

        "baseline_mae":
            baseline_mae,

        "audio_mae":
            model_mae,

        "mae_improvement":
            improvement,

        "spearman":
            rho
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
# PERMUTATION P-VALUES
# ============================================================

# +1 correction prevents a p-value of exactly zero.

mae_p = (
    1
    +
    np.sum(
        results[
            "mae_improvement"
        ]
        >= observed_improvement
    )
) / (
    N_PERMUTATIONS
    + 1
)


spearman_p = (
    1
    +
    np.sum(
        results["spearman"]
        >= observed_spearman
    )
) / (
    N_PERMUTATIONS
    + 1
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION TEST RESULTS")
print("=" * 70)


print("\nObserved audio improvement:")

print(
    f"{observed_improvement:.6f}"
)


print("\nNull improvement distribution:")

print(
    results[
        "mae_improvement"
    ].describe()
)


print(
    "\nPermutation p-value "
    "(MAE improvement):",
    mae_p
)


print(
    "\nObserved Spearman:",
    observed_spearman
)


print(
    "Permutation p-value "
    "(Spearman):",
    spearman_p
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# GRAPH:
# NULL DISTRIBUTION OF MAE IMPROVEMENT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    results["mae_improvement"],
    bins=25,
    edgecolor="black"
)

plt.axvline(
    observed_improvement,
    linestyle="--",
    linewidth=2,
    label=(
        "Observed improvement = "
        f"{observed_improvement:.4f}"
    )
)

plt.xlabel(
    "MAE improvement over mean baseline"
)

plt.ylabel(
    "Number of shuffled datasets"
)

plt.title(
    "Permutation Test: "
    "Does Audio Predict Listener Disagreement?"
)

plt.legend()

plt.tight_layout()


plot_path = os.path.join(
    DATA_FOLDER,
    "audio_permutation_test.png"
)


plt.savefig(
    plot_path,
    dpi=300
)

plt.show()


print("\nSaved:")
print(OUTPUT_FILE)
print(plot_path)