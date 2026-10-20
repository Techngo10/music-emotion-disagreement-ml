# Predicting Inter-Listener Emotional Disagreement in Music

Machine learning project investigating whether acoustic characteristics
of music can predict how strongly listeners disagree about a song's
emotional response.

The project uses the MERP dataset and derives a song-level disagreement
target from continuous valence and arousal annotations.

## Current Findings

- Acoustic features predict listener disagreement better than a mean baseline.
- Audio-only prediction was statistically significant under permutation testing.
- Mean valence and arousal provide substantially stronger predictive information.
- Adding acoustic features after controlling for mean emotion did not reliably
  improve predictive accuracy.

## Dataset

MERP (Music Emotion Recognition with Profile Information).

Raw dataset files and audio are not included in this repository.

## Project Structure

- `scripts/` — main data preparation and modelling pipeline
- `experiments/` — additional analyses and rejected/alternative approaches
- `src/` — reusable project code
- `results/` — final tables and figures
- `data/` — local dataset and generated modelling data

## Main Modelling Approach

The final evaluation uses nested cross-validation with:

- StandardScaler
- PCA for high-dimensional acoustic features
- Ridge regression
- MAE as the primary evaluation metric
- R² and Spearman correlation as secondary metrics
- permutation testing
- repeated nested cross-validation