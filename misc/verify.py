import os

folder = "data/MERP/audio"

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(".wav")
]

print("WAV files:", len(files))

print(files[:10])