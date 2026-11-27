
import pandas as pd
from pathlib import Path
from eeg_spectrogram_test import (
    spectrogram_from_eeg,
    extract_eeg_features,
)

# Mappen
EEG_PATH = Path("data/hms/train_eegs")
OUTPUT_PATH = Path("eeg_features_train.csv")
ERROR_PATH = Path("eeg_feature_errors.csv")
 
# Eerst 10 bestanden testen.
# Zet later op None om alle bestanden te verwerken.
MAX_FILES = None

# Per 100 EEG's resultaten opslaan
BATCH_SIZE = 100


def main():
    files = sorted(EEG_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"Geen EEG-bestanden gevonden in {EEG_PATH}"
        )

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Aantal EEG-bestanden om te verwerken: {len(files)}")

    # Oude outputbestanden verwijderen
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    if ERROR_PATH.exists():
        ERROR_PATH.unlink()

    batch = []
    errors = []
    processed = 0

    for i, file in enumerate(files, start=1):
        try:
            # Spectrogrammen berekenen
            img = spectrogram_from_eeg(file)

            # Numerieke features berekenen
            features = extract_eeg_features(img)

            # EEG-ID toevoegen
            features["eeg_id"] = int(file.stem)

            batch.append(features)
            processed += 1

        except Exception as e:
            errors.append({
                "eeg_id": file.stem,
                "error": str(e)
            })

        # Batch opslaan
        if len(batch) >= BATCH_SIZE or i == len(files):
            if batch:
                df = pd.DataFrame(batch)

                df.to_csv(
                    OUTPUT_PATH,
                    mode="a",
                    header=not OUTPUT_PATH.exists(),
                    index=False
                )

                batch = []

            print(
                f"Voortgang: {i}/{len(files)} "
                f"| Succesvol: {processed} "
                f"| Fouten: {len(errors)}"
            )

    # Eventuele fouten opslaan
    if errors:
        pd.DataFrame(errors).to_csv(
            ERROR_PATH,
            index=False
        )

    print("\nVerwerking afgerond.")
    print("Succesvol verwerkt:", processed)
    print("Aantal fouten:", len(errors))
    print("Resultaat:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
