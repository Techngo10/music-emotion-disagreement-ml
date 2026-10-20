import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data/MERP"

INPUT_FILE = os.path.join(
    DATA_FOLDER,
    "modelling_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "nested_cv_predictions.csv"
)

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("NESTED CROSS-VALIDATION")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# TARGET + CONTROLS
# ============================================================

TARGET = "overall_disagreement"

CONTROL_COLUMNS = [
    "mean_valence",
    "mean_arousal",
    "n_raters"
]


# ============================================================
# IDENTIFY AUDIO FEATURES
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

print("Audio features:", len(AUDIO_COLUMNS))
print("Control features:", CONTROL_COLUMNS)


# ============================================================
# DATA
# ============================================================

y = df[TARGET].to_numpy()

X_controls = df[
    CONTROL_COLUMNS
].to_numpy()

X_audio = df[
    AUDIO_COLUMNS
].to_numpy()

X_audio_controls = df[
    AUDIO_COLUMNS + CONTROL_COLUMNS
].to_numpy()


# ============================================================
# CV STRUCTURE
# ============================================================
#
# OUTER CV:
#   Gives unbiased held-out predictions.
#
# INNER CV:
#   Selects hyperparameters using training data ONLY.
#
# This prevents tuning on the outer test fold.
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


# ============================================================
# MODEL DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# CONTROLS-ONLY RIDGE
# ------------------------------------------------------------

control_pipeline = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "model",
        Ridge()
    )
])

control_grid = {
    "model__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


# ------------------------------------------------------------
# AUDIO RIDGE
# ------------------------------------------------------------

audio_ridge_pipeline = Pipeline([
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

audio_ridge_grid = {
    # Percentage of variance retained
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


# ------------------------------------------------------------
# AUDIO SVR
# ------------------------------------------------------------

audio_svr_pipeline = Pipeline([
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
        SVR(
            kernel="rbf"
        )
    )
])

audio_svr_grid = {

    "pca__n_components": [
        0.80,
        0.90,
        0.95
    ],

    "model__C": [
        0.1,
        1,
        10,
        100
    ],

    "model__gamma": [
        "scale",
        0.01,
        0.1
    ],

    "model__epsilon": [
        0.005,
        0.01,
        0.02,
        0.05
    ]
}


# ------------------------------------------------------------
# AUDIO + CONTROLS SVR
#
# For this first test we place all predictors in the same
# standardisation/PCA pipeline.
#
# Later, if this looks promising, we can use a ColumnTransformer
# to preserve the controls separately from the PCA audio space.
# ------------------------------------------------------------

audio_control_svr_pipeline = Pipeline([
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
        SVR(
            kernel="rbf"
        )
    )
])

audio_control_svr_grid = {

    "pca__n_components": [
        0.80,
        0.90,
        0.95
    ],

    "model__C": [
        0.1,
        1,
        10,
        100
    ],

    "model__gamma": [
        "scale",
        0.01,
        0.1
    ],

    "model__epsilon": [
        0.005,
        0.01,
        0.02,
        0.05
    ]
}


# ============================================================
# STORAGE FOR OUTER-FOLD PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "song_id": df["song_id"],
    "actual": y,

    "mean_baseline":
        np.nan,

    "controls_ridge":
        np.nan,

    "audio_ridge":
        np.nan,

    "audio_svr":
        np.nan,

    "audio_controls_svr":
        np.nan
})


# ============================================================
# NESTED CV
# ============================================================

print("\nRunning 5-fold outer CV...")
print("Hyperparameters are tuned inside each training fold.")


for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(df),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {fold}/5")
    print("=" * 70)

    y_train = y[train_idx]


    # ========================================================
    # 1. MEAN BASELINE
    # ========================================================

    training_mean = np.mean(y_train)

    predictions.loc[
        test_idx,
        "mean_baseline"
    ] = training_mean


    # ========================================================
    # 2. CONTROLS-ONLY RIDGE
    # ========================================================

    search_controls = GridSearchCV(
        estimator=control_pipeline,
        param_grid=control_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search_controls.fit(
        X_controls[train_idx],
        y_train
    )

    predictions.loc[
        test_idx,
        "controls_ridge"
    ] = search_controls.predict(
        X_controls[test_idx]
    )

    print(
        "Controls Ridge:",
        search_controls.best_params_
    )


    # ========================================================
    # 3. AUDIO RIDGE
    # ========================================================

    search_audio_ridge = GridSearchCV(
        estimator=audio_ridge_pipeline,
        param_grid=audio_ridge_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search_audio_ridge.fit(
        X_audio[train_idx],
        y_train
    )

    predictions.loc[
        test_idx,
        "audio_ridge"
    ] = search_audio_ridge.predict(
        X_audio[test_idx]
    )

    print(
        "Audio Ridge:",
        search_audio_ridge.best_params_
    )


    # ========================================================
    # 4. AUDIO SVR
    # ========================================================

    search_audio_svr = GridSearchCV(
        estimator=audio_svr_pipeline,
        param_grid=audio_svr_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search_audio_svr.fit(
        X_audio[train_idx],
        y_train
    )

    predictions.loc[
        test_idx,
        "audio_svr"
    ] = search_audio_svr.predict(
        X_audio[test_idx]
    )

    print(
        "Audio SVR:",
        search_audio_svr.best_params_
    )


    # ========================================================
    # 5. AUDIO + CONTROLS SVR
    # ========================================================

    search_audio_controls = GridSearchCV(
        estimator=audio_control_svr_pipeline,
        param_grid=audio_control_svr_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search_audio_controls.fit(
        X_audio_controls[train_idx],
        y_train
    )

    predictions.loc[
        test_idx,
        "audio_controls_svr"
    ] = search_audio_controls.predict(
        X_audio_controls[test_idx]
    )

    print(
        "Audio + Controls SVR:",
        search_audio_controls.best_params_
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    spearman, spearman_p = spearmanr(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman,
        "Spearman_p": spearman_p
    }


models = [
    "mean_baseline",
    "controls_ridge",
    "audio_ridge",
    "audio_svr",
    "audio_controls_svr"
]

results = []


print("\n" + "=" * 70)
print("OUT-OF-SAMPLE RESULTS")
print("=" * 70)


for model in models:

    metrics = evaluate(
        predictions["actual"],
        predictions[model]
    )

    results.append({
        "model": model,
        **metrics
    })


results_df = pd.DataFrame(
    results
)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# IMPROVEMENT OVER MEAN BASELINE
# ============================================================

baseline_mae = results_df.loc[
    results_df["model"] == "mean_baseline",
    "MAE"
].iloc[0]


print("\n" + "=" * 70)
print("MAE IMPROVEMENT OVER MEAN BASELINE")
print("=" * 70)


for _, row in results_df.iterrows():

    improvement = (
        baseline_mae
        - row["MAE"]
    )

    percentage = (
        improvement
        / baseline_mae
        * 100
    )

    print(
        f"{row['model']:22s} "
        f"MAE={row['MAE']:.4f} "
        f"Improvement={improvement:+.4f} "
        f"({percentage:+.1f}%)"
    )


# ============================================================
# SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_FILE,
    index=False
)

results_path = os.path.join(
    DATA_FOLDER,
    "nested_cv_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)
print(results_path)