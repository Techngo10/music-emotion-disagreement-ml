from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "MERP" / "audio"

files = sorted(p.name for p in AUDIO_DIR.glob("*.wav"))

print("WAV files:", len(files))
print(files[:10])
