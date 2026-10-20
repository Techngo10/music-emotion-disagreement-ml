import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

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

OUTPUT_PREDICTIONS = os.path.join(
    DATA_FOLDER,
    "corrected_combined_predictions.csv"
)

OUTPUT_RESULTS = os.path.join(
    DATA_FOLDER,
    "corrected_combined_results.csv"
)

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

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
    c for c in df.columns
    if c not in non_audio_columns
]

ALL_COLUMNS = (
    AUDIO_COLUMNS
    + CONTROL_COLUMNS
)

print("=" * 70)
print("CORRECTED AUDIO + CONTROLS ANALYSIS")
print("=" * 70)

print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Controls:", CONTROL_COLUMNS)


# ============================================================
# DATA
# ============================================================

X = df[ALL_COLUMNS]
X_controls = df[CONTROL_COLUMNS]

y = df[TARGET].to_numpy()


# ============================================================
# OUTER AND INNER CV
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
# PREPROCESSING
#
# AUDIO:
#   standardise -> PCA
#
# CONTROLS:
#   standardise only
#
# The controls are NOT included in PCA.
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


combined_preprocessor = ColumnTransformer([
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


# ============================================================
# MODEL 1:
# CONTROLS-ONLY RIDGE
# ============================================================

controls_ridge = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "model",
        Ridge()
    )
])

controls_ridge_grid = {
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
# MODEL 2:
# CONTROLS-ONLY SVR
#
# Useful because later we compare it against
# Audio + Controls SVR using the SAME model family.
# ============================================================

controls_svr = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "model",
        SVR(
            kernel="rbf"
        )
    )
])

controls_svr_grid = {

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
# MODEL 3:
# AUDIO + CONTROLS RIDGE
# ============================================================

combined_ridge = Pipeline([
    (
        "preprocess",
        combined_preprocessor
    ),
    (
        "model",
        Ridge()
    )
])

combined_ridge_grid = {

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
# MODEL 4:
# AUDIO + CONTROLS SVR
# ============================================================

combined_svr = Pipeline([
    (
        "preprocess",
        combined_preprocessor
    ),
    (
        "model",
        SVR(
            kernel="rbf"
        )
    )
])

combined_svr_grid = {

    "preprocess__audio__pca__n_components": [
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
# STORAGE
# ============================================================

predictions = pd.DataFrame({
    "song_id": df["song_id"],
    "actual": y,

    "mean_baseline":
        np.nan,

    "controls_ridge":
        np.nan,

    "controls_svr":
        np.nan,

    "audio_controls_ridge":
        np.nan,

    "audio_controls_svr":
        np.nan
})


# ============================================================
# NESTED CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(df),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {fold}/5")
    print("=" * 70)

    y_train = y[train_idx]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    controls_train = X_controls.iloc[
        train_idx
    ]

    controls_test = X_controls.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # MEAN BASELINE
    # --------------------------------------------------------

    predictions.loc[
        test_idx,
        "mean_baseline"
    ] = np.mean(y_train)


    # --------------------------------------------------------
    # CONTROLS RIDGE
    # --------------------------------------------------------

    search = GridSearchCV(
        controls_ridge,
        controls_ridge_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search.fit(
        controls_train,
        y_train
    )

    predictions.loc[
        test_idx,
        "controls_ridge"
    ] = search.predict(
        controls_test
    )

    print(
        "Controls Ridge:",
        search.best_params_
    )


    # --------------------------------------------------------
    # CONTROLS SVR
    # --------------------------------------------------------

    search = GridSearchCV(
        controls_svr,
        controls_svr_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search.fit(
        controls_train,
        y_train
    )

    predictions.loc[
        test_idx,
        "controls_svr"
    ] = search.predict(
        controls_test
    )

    print(
        "Controls SVR:",
        search.best_params_
    )


    # --------------------------------------------------------
    # AUDIO + CONTROLS RIDGE
    # --------------------------------------------------------

    search = GridSearchCV(
        combined_ridge,
        combined_ridge_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    predictions.loc[
        test_idx,
        "audio_controls_ridge"
    ] = search.predict(
        X_test
    )

    print(
        "Audio + Controls Ridge:",
        search.best_params_
    )


    # --------------------------------------------------------
    # AUDIO + CONTROLS SVR
    # --------------------------------------------------------

    search = GridSearchCV(
        combined_svr,
        combined_svr_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    predictions.loc[
        test_idx,
        "audio_controls_svr"
    ] = search.predict(
        X_test
    )

    print(
        "Audio + Controls SVR:",
        search.best_params_
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(actual, predicted):

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

    rho, p = spearmanr(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": rho,
        "Spearman_p": p
    }


# ============================================================
# RESULTS
# ============================================================

models = [
    "mean_baseline",
    "controls_ridge",
    "controls_svr",
    "audio_controls_ridge",
    "audio_controls_svr"
]

results = []


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


print("\n" + "=" * 70)
print("CORRECTED OUT-OF-SAMPLE RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# AUDIO ADDED VALUE
# ============================================================

print("\n" + "=" * 70)
print("DO AUDIO FEATURES ADD VALUE?")
print("=" * 70)


def compare_models(
    base_model,
    combined_model
):

    base_mae = results_df.loc[
        results_df["model"] == base_model,
        "MAE"
    ].iloc[0]

    combined_mae = results_df.loc[
        results_df["model"] == combined_model,
        "MAE"
    ].iloc[0]

    improvement = (
        base_mae
        - combined_mae
    )

    percent = (
        improvement
        / base_mae
        * 100
    )

    print(
        f"\n{base_model}"
        f" -> {combined_model}"
    )

    print(
        f"Base MAE:     {base_mae:.4f}"
    )

    print(
        f"Combined MAE: {combined_mae:.4f}"
    )

    print(
        f"Difference:   {improvement:+.4f}"
    )

    print(
        f"Change:       {percent:+.1f}%"
    )


compare_models(
    "controls_ridge",
    "audio_controls_ridge"
)

compare_models(
    "controls_svr",
    "audio_controls_svr"
)


# ============================================================
# SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

print("\nSaved:")
print(OUTPUT_PREDICTIONS)
print(OUTPUT_RESULTS)