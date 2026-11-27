import pandas as pd
from pathlib import Path

eeg_path = Path("data/hms/train_eegs")

files = list(eeg_path.glob("*.parquet"))
print("Aantal EEG-bestanden:", len(files))

eeg = pd.read_parquet(files[0])

print("Voorbeeldbestand:", files[0].name)
print("Shape:", eeg.shape)
print("Kolommen:")
print(eeg.columns.tolist())
print("Eerste rijen:")
print(eeg.head())