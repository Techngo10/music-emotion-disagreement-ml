"""Convenience runner for the main MERP disagreement pipeline.

The project remains split into focused scripts. This file simply provides a
single entry point when you want to run the stages in order.
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

CORE_STEPS = [
    "01_prepare_target.py",
    "02_target_diagnostics.py",
    "03_split_half_reliability.py",
    "05_extract_audio_features.py",
    "06_run_nested_models.py",
]

ROBUSTNESS_STEPS = [
    "07_cv_stability.py",
    "08_audio_permutation_test.py",
    "09_incremental_audio_test.py",
]


def run_script(filename: str) -> None:
    script = SCRIPTS_DIR / filename
    print("\n" + "=" * 70)
    print(f"RUNNING {filename}")
    print("=" * 70)
    subprocess.run([sys.executable, str(script)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download the 50 MERP audio files before feature extraction.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Also run repeated-CV and permutation robustness tests.",
    )
    args = parser.parse_args()

    steps = CORE_STEPS.copy()

    if args.download_audio:
        steps.insert(3, "04_download_audio.py")

    if args.robustness:
        steps.extend(ROBUSTNESS_STEPS)

    for step in steps:
        run_script(step)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
