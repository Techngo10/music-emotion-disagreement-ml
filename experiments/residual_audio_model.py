import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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
    "residual_audio_predictions.csv"
)

RESULTS_FILE = os.path.join(
    DATA_FOLDER,
    "residual_audio_results.csv"
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
    column
    for column in df.columns
    if column not in non_audio_columns
]


X_controls = df[
    CONTROL_COLUMNS
].to_numpy()

X_audio = df[
    AUDIO_COLUMNS
].to_numpy()

y = df[
    TARGET
].to_numpy()


print("=" * 70)
print("RESIDUAL AUDIO ANALYSIS")
print("=" * 70)

print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Controls:", CONTROL_COLUMNS)


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


# ============================================================
# CONTROL MODEL
# ============================================================

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


# ============================================================
# AUDIO RESIDUAL MODEL
# ============================================================

audio_pipeline = Pipeline([
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


audio_grid = {

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
# STORAGE
# ============================================================

predictions = pd.DataFrame({
    "song_id":
        df["song_id"],

    "actual":
        y,

    "control_prediction":
        np.nan,

    "actual_residual":
        np.nan,

    "audio_residual_prediction":
        np.nan,

    "control_plus_audio_prediction":
        np.nan
})


# ============================================================
# OUTER CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(df),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {fold}/5")
    print("=" * 70)

    Xc_train = X_controls[
        train_idx
    ]

    Xc_test = X_controls[
        test_idx
    ]

    Xa_train = X_audio[
        train_idx
    ]

    Xa_test = X_audio[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]


    # ========================================================
    # STEP 1:
    # Tune controls model using outer-training data only
    # ========================================================

    control_search = GridSearchCV(

        estimator=control_pipeline,

        param_grid=control_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,

        n_jobs=-1
    )


    control_search.fit(
        Xc_train,
        y_train
    )


    print(
        "Control parameters:",
        control_search.best_params_
    )


    # ========================================================
    # STEP 2:
    # Control prediction for OUTER TEST data
    # ========================================================

    control_test_prediction = (
        control_search.predict(
            Xc_test
        )
    )


    # ========================================================
    # STEP 3:
    # Generate out-of-fold control predictions for
    # TRAINING songs.
    #
    # This prevents us from calculating residuals using
    # predictions fitted directly to the same observations.
    # ========================================================

    best_control_model = clone(
        control_search.best_estimator_
    )


    control_train_oof = cross_val_predict(

        best_control_model,

        Xc_train,

        y_train,

        cv=inner_cv,

        n_jobs=-1
    )


    # ========================================================
    # STEP 4:
    # Training residuals
    # ========================================================

    residual_train = (
        y_train
        - control_train_oof
    )


    print(
        "Training residual SD:",
        np.std(residual_train)
    )


    # ========================================================
    # STEP 5:
    # Tune audio model to predict CONTROL RESIDUALS
    # ========================================================

    audio_search = GridSearchCV(

        estimator=audio_pipeline,

        param_grid=audio_grid,

        scoring="neg_mean_absolute_error",

        cv=inner_cv,

        n_jobs=-1
    )


    audio_search.fit(
        Xa_train,
        residual_train
    )


    print(
        "Residual audio parameters:",
        audio_search.best_params_
    )


    # ========================================================
    # STEP 6:
    # Predict residual for unseen songs
    # ========================================================

    residual_test_prediction = (
        audio_search.predict(
            Xa_test
        )
    )


    # ========================================================
    # ACTUAL TEST RESIDUAL
    #
    # How wrong was the control model?
    # ========================================================

    actual_test_residual = (
        y_test
        - control_test_prediction
    )


    # ========================================================
    # FINAL PREDICTION
    #
    # Controls prediction
    # +
    # Audio correction
    # ========================================================

    corrected_prediction = (
        control_test_prediction
        +
        residual_test_prediction
    )


    # ========================================================
    # STORE
    # ========================================================

    predictions.loc[
        test_idx,
        "control_prediction"
    ] = control_test_prediction


    predictions.loc[
        test_idx,
        "actual_residual"
    ] = actual_test_residual


    predictions.loc[
        test_idx,
        "audio_residual_prediction"
    ] = residual_test_prediction


    predictions.loc[
        test_idx,
        "control_plus_audio_prediction"
    ] = corrected_prediction


# ============================================================
# METRICS
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
# 1. ORIGINAL TARGET
# ============================================================

control_metrics = evaluate(

    predictions[
        "actual"
    ],

    predictions[
        "control_prediction"
    ]
)


corrected_metrics = evaluate(

    predictions[
        "actual"
    ],

    predictions[
        "control_plus_audio_prediction"
    ]
)


# ============================================================
# 2. RESIDUAL TARGET
#
# Zero prediction means:
# "Audio provides no correction beyond controls."
# ============================================================

actual_residual = predictions[
    "actual_residual"
].to_numpy()

predicted_residual = predictions[
    "audio_residual_prediction"
].to_numpy()


zero_residual_prediction = np.zeros(
    len(df)
)


zero_residual_metrics = evaluate(
    actual_residual,
    zero_residual_prediction
)


audio_residual_metrics = evaluate(
    actual_residual,
    predicted_residual
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame([

    {
        "model":
            "controls_only",

        **control_metrics
    },

    {
        "model":
            "controls_plus_audio_residual",

        **corrected_metrics
    },

    {
        "model":
            "zero_residual_baseline",

        **zero_residual_metrics
    },

    {
        "model":
            "audio_predicting_residual",

        **audio_residual_metrics
    }
])


print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS RESULTS")
print("=" * 70)


print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# ADDED VALUE
# ============================================================

control_mae = (
    control_metrics["MAE"]
)

corrected_mae = (
    corrected_metrics["MAE"]
)


mae_gain = (
    control_mae
    - corrected_mae
)


mae_percent = (
    mae_gain
    / control_mae
    * 100
)


print("\n" + "=" * 70)
print("AUDIO RESIDUAL CORRECTION")
print("=" * 70)


print(
    f"Controls-only MAE: "
    f"{control_mae:.6f}"
)


print(
    f"Controls + audio correction MAE: "
    f"{corrected_mae:.6f}"
)


print(
    f"MAE improvement: "
    f"{mae_gain:+.6f}"
)


print(
    f"Percentage change: "
    f"{mae_percent:+.2f}%"
)


print(
    "\nResidual Spearman:"
)


print(
    f"{audio_residual_metrics['Spearman']:.4f}"
)


# ============================================================
# SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_FILE,
    index=False
)

results.to_csv(
    RESULTS_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)
print(RESULTS_FILE)