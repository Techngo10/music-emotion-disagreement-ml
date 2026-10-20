from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MERP"
PROCESSED_DIR = DATA_DIR / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = PROCESSED_DIR / "modelling_dataset.csv"
OUTPUT_PREDICTIONS = TABLES_DIR / "nested_cv_predictions.csv"
OUTPUT_RESULTS = TABLES_DIR / "nested_cv_results.csv"

TARGET = "overall_disagreement"
CONTROL_COLUMNS = [
    "mean_valence",
    "mean_arousal",
    "n_raters",
]

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

NON_AUDIO_COLUMNS = {
    "song_id",
    "n_raters",
    "mean_valence",
    "mean_arousal",
    "valence_extremity",
    "arousal_extremity",
    "valence_disagreement",
    "arousal_disagreement",
    "overall_disagreement",
}

AUDIO_COLUMNS = [
    column
    for column in df.columns
    if column not in NON_AUDIO_COLUMNS
]

X_audio = df[AUDIO_COLUMNS]
X_controls = df[CONTROL_COLUMNS]
X_combined = df[AUDIO_COLUMNS + CONTROL_COLUMNS]
y = df[TARGET].to_numpy()

print("=" * 70)
print("FINAL NESTED CROSS-VALIDATION")
print("=" * 70)
print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Controls:", CONTROL_COLUMNS)


# ============================================================
# CROSS-VALIDATION
# ============================================================

outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED,
)

inner_cv = KFold(
    n_splits=4,
    shuffle=True,
    random_state=RANDOM_SEED,
)


# ============================================================
# MODEL PIPELINES
# ============================================================

# Audio-only: scale -> PCA -> Ridge
audio_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA()),
    ("model", Ridge()),
])

audio_ridge_grid = {
    "pca__n_components": [0.80, 0.90, 0.95],
    "model__alpha": [0.01, 0.1, 1, 10, 100],
}

# Controls-only: scale -> Ridge
controls_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge()),
])

controls_ridge_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
}

# Combined model: PCA is applied ONLY to audio.
audio_preprocessing = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA()),
])

combined_preprocessor = ColumnTransformer([
    (
        "audio",
        audio_preprocessing,
        AUDIO_COLUMNS,
    ),
    (
        "controls",
        StandardScaler(),
        CONTROL_COLUMNS,
    ),
])

combined_ridge = Pipeline([
    ("preprocess", combined_preprocessor),
    ("model", Ridge()),
])

combined_ridge_grid = {
    "preprocess__audio__pca__n_components": [0.80, 0.90, 0.95],
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
}


# ============================================================
# STORAGE
# ============================================================

predictions = pd.DataFrame({
    "song_id": df["song_id"],
    "actual": y,
    "mean_baseline": np.nan,
    "audio_ridge": np.nan,
    "controls_ridge": np.nan,
    "audio_controls_ridge": np.nan,
})


# ============================================================
# NESTED CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(df),
    start=1,
):
    print("\n" + "=" * 70)
    print(f"OUTER FOLD {fold}/5")
    print("=" * 70)

    y_train = y[train_idx]

    # --------------------------------------------------------
    # Mean baseline
    # --------------------------------------------------------
    predictions.loc[test_idx, "mean_baseline"] = np.mean(y_train)

    # --------------------------------------------------------
    # Audio-only Ridge
    # --------------------------------------------------------
    audio_search = GridSearchCV(
        estimator=audio_ridge,
        param_grid=audio_ridge_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
    )

    audio_search.fit(
        X_audio.iloc[train_idx],
        y_train,
    )

    predictions.loc[test_idx, "audio_ridge"] = audio_search.predict(
        X_audio.iloc[test_idx]
    )

    print("Audio Ridge:", audio_search.best_params_)

    # --------------------------------------------------------
    # Controls-only Ridge
    # --------------------------------------------------------
    controls_search = GridSearchCV(
        estimator=controls_ridge,
        param_grid=controls_ridge_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
    )

    controls_search.fit(
        X_controls.iloc[train_idx],
        y_train,
    )

    predictions.loc[test_idx, "controls_ridge"] = controls_search.predict(
        X_controls.iloc[test_idx]
    )

    print("Controls Ridge:", controls_search.best_params_)

    # --------------------------------------------------------
    # Audio + controls Ridge
    # --------------------------------------------------------
    combined_search = GridSearchCV(
        estimator=combined_ridge,
        param_grid=combined_ridge_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
    )

    combined_search.fit(
        X_combined.iloc[train_idx],
        y_train,
    )

    predictions.loc[test_idx, "audio_controls_ridge"] = combined_search.predict(
        X_combined.iloc[test_idx]
    )

    print("Audio + Controls Ridge:", combined_search.best_params_)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(actual, predicted):
    """Return the project evaluation metrics for one set of predictions."""

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    rho, p_value = spearmanr(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": rho,
        "Spearman_p": p_value,
    }


MODEL_COLUMNS = [
    "mean_baseline",
    "audio_ridge",
    "controls_ridge",
    "audio_controls_ridge",
]

results = []

for model_name in MODEL_COLUMNS:
    metrics = evaluate(
        predictions["actual"],
        predictions[model_name],
    )

    results.append({
        "model": model_name,
        **metrics,
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("OUT-OF-SAMPLE RESULTS")
print("=" * 70)
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# ============================================================
# KEY COMPARISONS
# ============================================================

mean_mae = results_df.loc[
    results_df["model"] == "mean_baseline",
    "MAE",
].iloc[0]

audio_mae = results_df.loc[
    results_df["model"] == "audio_ridge",
    "MAE",
].iloc[0]

controls_mae = results_df.loc[
    results_df["model"] == "controls_ridge",
    "MAE",
].iloc[0]

combined_mae = results_df.loc[
    results_df["model"] == "audio_controls_ridge",
    "MAE",
].iloc[0]

print("\n" + "=" * 70)
print("KEY MAE COMPARISONS")
print("=" * 70)

print(
    "Audio vs mean baseline: "
    f"{mean_mae - audio_mae:+.6f} "
    f"({(mean_mae - audio_mae) / mean_mae * 100:+.1f}%)"
)

print(
    "Audio + controls vs controls: "
    f"{controls_mae - combined_mae:+.6f} "
    f"({(controls_mae - combined_mae) / controls_mae * 100:+.1f}%)"
)


# ============================================================
# SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False,
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
)

print("\nSaved:")
print(OUTPUT_PREDICTIONS)
print(OUTPUT_RESULTS)
