# Predicting Inter-Listener Emotional Disagreement in Music

Machine-learning project investigating whether acoustic characteristics of music can predict how strongly listeners disagree about a song's emotional response.

The project uses the MERP dataset and constructs a song-level disagreement target from continuous listener valence and arousal annotations. The final modelling setup uses 50 non-anchor songs and a 260-feature openSMILE/IS13-derived acoustic representation.

## Current findings

- Audio-only Ridge regression consistently beats a mean-prediction baseline.
- The audio-only improvement is supported by permutation testing.
- Mean valence, mean arousal, and rater count provide substantially stronger predictive performance than audio alone.
- Adding audio after these controls does not reliably improve MAE.
- Repeated nested cross-validation shows audio beats the mean baseline in 10/10 repeats, while audio + controls beats controls alone in only 1/10 repeats.

The current interpretation is that acoustic characteristics contain genuine information about listener disagreement, but much of that predictive information overlaps with the mean emotional response to the song.

## Project structure

```text
.
├── run_pipeline.py
├── scripts/                 # Main reproducible analysis pipeline
├── experiments/             # Alternative / rejected analyses
├── exploratory/             # Early dataset inspection and visualisation
├── data/
│   └── MERP/
│       ├── processed/       # Generated modelling-ready CSV files
│       ├── audio/           # Local WAV files (not committed)
│       └── *.parquet        # MERP source files (not committed)
└── results/
    ├── tables/              # Model, reliability and permutation outputs
    └── figures/             # Generated plots
```

## Main scripts

1. `01_prepare_target.py` — inspect MERP data and construct the disagreement target.
2. `02_target_diagnostics.py` — analyse confounds and compare target reliability.
3. `03_split_half_reliability.py` — estimate stability across random listener halves.
4. `04_download_audio.py` — download the 50 non-anchor MERP audio files.
5. `05_extract_audio_features.py` — extract the 260-dimensional acoustic representation.
6. `06_run_nested_models.py` — final nested-CV comparison: mean baseline, audio Ridge, controls Ridge, and audio + controls Ridge.
7. `07_cv_stability.py` — repeat nested CV over multiple random fold assignments.
8. `08_audio_permutation_test.py` — test whether audio-only prediction exceeds chance.
9. `09_incremental_audio_test.py` — test whether correctly matched audio adds information beyond controls.

## Running the project

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run individual stages from the project root, for example:

```powershell
py scripts\06_run_nested_models.py
```

Or use the convenience runner after the MERP source data and audio are available:

```powershell
py run_pipeline.py
```

Include robustness tests:

```powershell
py run_pipeline.py --robustness
```

Download audio as part of the run:

```powershell
py run_pipeline.py --download-audio --robustness
```

## Evaluation

Hyperparameters are tuned inside an inner 4-fold cross-validation loop, while performance is estimated on an outer 5-fold cross-validation loop. Preprocessing is contained inside scikit-learn pipelines to reduce information leakage.

Primary metric: MAE. Secondary metrics: RMSE, R² and Spearman rank correlation. Robustness is assessed using repeated nested CV and permutation tests.

## Dataset

MERP (Music Emotion Recognition with Profile Information). Raw `.parquet` files and WAV audio are intentionally excluded from GitHub and must be downloaded locally.

## Notes

The acoustic representation is described as openSMILE/IS13-derived. The project recreates a 260-feature representation from 65 low-level descriptors using summary statistics; it should not be presented as an exact copy of MERP's original precomputed feature file unless that equivalence is separately verified.
