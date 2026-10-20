import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV

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
    "cv_stability_results.csv"
)

N_REPEATS = 10


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

TARGET = "overall_disagreement"

CONTROL_COLUMNS = [
    "mean_valence",
    "mean_arousal",
    "n_raters"
]


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


y = df[TARGET].to_numpy()

X_controls = df[
    CONTROL_COLUMNS
].copy()

X_audio = df[
    AUDIO_COLUMNS
].copy()

X_combined = df[
    AUDIO_COLUMNS + CONTROL_COLUMNS
].copy()


print("=" * 70)
print("REPEATED NESTED CV STABILITY ANALYSIS")
print("=" * 70)

print("Songs:", len(df))
print("Audio features:", len(AUDIO_COLUMNS))
print("Repeats:", N_REPEATS)


# ============================================================
# MODELS
# ============================================================

controls_pipeline = Pipeline([
    (
        "scale",
        StandardScaler()
    ),
    (
        "model",
        Ridge()
    )
])

controls_grid = {
    "model__alpha": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


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


audio_preprocessing = Pipeline([
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
        audio_preprocessing,
        AUDIO_COLUMNS
    ),

    (
        "controls",
        StandardScaler(),
        CONTROL_COLUMNS
    )
])


combined_pipeline = Pipeline([
    (
        "preprocess",
        combined_preprocessor
    ),
    (
        "model",
        Ridge()
    )
])


combined_grid = {
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
# METRICS
# ============================================================

def calculate_metrics(
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

    rho, _ = spearmanr(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": rho
    }


# ============================================================
# REPEATED NESTED CV
# ============================================================

all_results = []


for repeat in range(
    N_REPEATS
):

    seed = 42 + repeat

    print(
        f"\nRepeat "
        f"{repeat + 1}/{N_REPEATS} "
        f"(seed={seed})"
    )


    outer_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )


    inner_cv = KFold(
        n_splits=4,
        shuffle=True,
        random_state=seed
    )


    predictions = {

        "mean":
            np.zeros(len(df)),

        "controls":
            np.zeros(len(df)),

        "audio":
            np.zeros(len(df)),

        "combined":
            np.zeros(len(df))
    }


    for train_idx, test_idx in outer_cv.split(df):

        y_train = y[
            train_idx
        ]


        # ====================================================
        # MEAN BASELINE
        # ====================================================

        predictions["mean"][
            test_idx
        ] = np.mean(
            y_train
        )


        # ====================================================
        # CONTROLS RIDGE
        # ====================================================

        search = GridSearchCV(
            controls_pipeline,
            controls_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1
        )

        search.fit(
            X_controls.iloc[
                train_idx
            ],
            y_train
        )

        predictions["controls"][
            test_idx
        ] = search.predict(
            X_controls.iloc[
                test_idx
            ]
        )


        # ====================================================
        # AUDIO RIDGE
        # ====================================================

        search = GridSearchCV(
            audio_pipeline,
            audio_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1
        )

        search.fit(
            X_audio.iloc[
                train_idx
            ],
            y_train
        )

        predictions["audio"][
            test_idx
        ] = search.predict(
            X_audio.iloc[
                test_idx
            ]
        )


        # ====================================================
        # AUDIO + CONTROLS RIDGE
        # ====================================================

        search = GridSearchCV(
            combined_pipeline,
            combined_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1
        )

        search.fit(
            X_combined.iloc[
                train_idx
            ],
            y_train
        )

        predictions["combined"][
            test_idx
        ] = search.predict(
            X_combined.iloc[
                test_idx
            ]
        )


    # ========================================================
    # METRICS FOR THIS REPEAT
    # ========================================================

    for model_name in predictions:

        metrics = calculate_metrics(
            y,
            predictions[
                model_name
            ]
        )

        all_results.append({
            "repeat":
                repeat + 1,

            "seed":
                seed,

            "model":
                model_name,

            **metrics
        })


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("STABILITY SUMMARY")
print("=" * 70)


summary = (
    results
    .groupby("model")
    .agg({

        "MAE": [
            "mean",
            "std",
            "min",
            "max"
        ],

        "R2": [
            "mean",
            "std"
        ],

        "Spearman": [
            "mean",
            "std",
            "min",
            "max"
        ]
    })
)


print(summary)


# ============================================================
# HOW OFTEN DOES AUDIO BEAT BASELINE?
# ============================================================

pivot_mae = results.pivot(
    index="repeat",
    columns="model",
    values="MAE"
)


audio_beats_mean = (
    pivot_mae["audio"]
    <
    pivot_mae["mean"]
).sum()


combined_beats_controls = (
    pivot_mae["combined"]
    <
    pivot_mae["controls"]
).sum()


print("\n" + "=" * 70)
print("CONSISTENCY CHECK")
print("=" * 70)


print(
    f"Audio beats mean baseline: "
    f"{audio_beats_mean}/{N_REPEATS} repeats"
)


print(
    f"Audio + controls beats controls: "
    f"{combined_beats_controls}/{N_REPEATS} repeats"
)


# ============================================================
# SPEARMAN COMPARISON
# ============================================================

pivot_rho = results.pivot(
    index="repeat",
    columns="model",
    values="Spearman"
)


combined_ranks_better = (
    pivot_rho["combined"]
    >
    pivot_rho["controls"]
).sum()


print(
    f"Audio + controls has higher "
    f"Spearman than controls: "
    f"{combined_ranks_better}/{N_REPEATS} repeats"
)


print("\nSaved:")
print(OUTPUT_FILE)