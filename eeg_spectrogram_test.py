
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

# Vier EEG-groepen uit de Kaggle-notebook
NAMES = ["LL", "LP", "RP", "RL"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],  # LL
    ["Fp1", "F3", "C3", "P3", "O1"],  # LP
    ["Fp2", "F4", "C4", "P4", "O2"],  # RP
    ["Fp2", "F8", "T4", "T6", "O2"],  # RL
]


def spectrogram_from_eeg(parquet_path):
    # EEG inlezen
    eeg = pd.read_parquet(parquet_path)

    # Middelste 50 seconden selecteren
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    if len(eeg) != 10_000:
        raise ValueError(
            f"Verwacht 10.000 meetpunten, maar kreeg {len(eeg)}."
        )

    # Vier spectrogrammen opslaan
    img = np.zeros((128, 256, 4), dtype="float32")

    for k in range(4):
        cols = FEATS[k]

        for kk in range(4):
            # Bipolaire afleiding berekenen
            x = (
                eeg[cols[kk]].to_numpy(dtype=np.float32)
                - eeg[cols[kk + 1]].to_numpy(dtype=np.float32)
            )

            # NaN-waarden vervangen
            m = np.nanmean(x)

            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # Mel-spectrogram berekenen
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            # Omzetten naar decibel
            width = (mel_spec.shape[1] // 32) * 32

            mel_spec_db = librosa.power_to_db(
                mel_spec,
                ref=np.max
            ).astype(np.float32)[:, :width]

            # Normaliseren
            mel_spec_db = (mel_spec_db + 40) / 40

            # Spectrogram toevoegen aan de juiste groep
            img[:, :, k] += mel_spec_db

        # Gemiddelde van de vier afleidingen
        img[:, :, k] /= 4.0

    return img


def extract_eeg_features(img):
    """
    Zet de vier EEG-spectrogrammen om naar numerieke features.

    img: array met shape (128, 256, 4)
    """

    names = ["LL", "LP", "RP", "RL"]
    features = {}

    # 8 opeenvolgende groepen van 16 Mel-bins
    for k, region in enumerate(names):
        spec = img[:, :, k]

        for band in range(8):
            f_start = band * 16
            f_end = f_start + 16

            freq_region = spec[f_start:f_end, :]

            # Statistieken over frequentie en tijd
            features[f"{region}_mel{band}_mean"] = np.mean(freq_region)
            features[f"{region}_mel{band}_std"] = np.std(freq_region)
            features[f"{region}_mel{band}_min"] = np.min(freq_region)
            features[f"{region}_mel{band}_max"] = np.max(freq_region)

            # Gemiddelde activiteit in vier tijdssegmenten
            for t in range(4):
                t_start = t * 64
                t_end = t_start + 64

                time_region = freq_region[:, t_start:t_end]

                features[
                    f"{region}_mel{band}_time{t}_mean"
                ] = np.mean(time_region)

    # Verschillen tussen linker- en rechterhersenhelft
    for band in range(8):
        for stat in ["mean", "std"]:
            left = features[f"LL_mel{band}_{stat}"]
            right = features[f"RL_mel{band}_{stat}"]

            features[f"asym_lateral_mel{band}_{stat}"] = left - right

            left = features[f"LP_mel{band}_{stat}"]
            right = features[f"RP_mel{band}_{stat}"]

            features[f"asym_posterior_mel{band}_{stat}"] = left - right

    return features

if __name__ == "__main__":

    eeg_path = Path("data/hms/train_eegs")

    files = sorted(eeg_path.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"Geen EEG-bestanden gevonden in {eeg_path}")

    # Eerste EEG als test gebruiken
    test_file = files[0]

    print("Testbestand:", test_file.name)

    img = spectrogram_from_eeg(test_file)
    # Numerieke features berekenen
    features = extract_eeg_features(img)

    print("Aantal nieuwe EEG-features:", len(features))

    # Features bewaren voor inspectie
    feature_df = pd.DataFrame([features])
    feature_df.insert(0, "eeg_id", test_file.stem)

    feature_df.to_csv("eeg_features_test.csv", index=False)

    print("Features opgeslagen in eeg_features_test.csv")
    print(feature_df.iloc[:, :10])
    print("Shape van spectrogrammen:", img.shape)
    print("Minimum:", np.nanmin(img))
    print("Maximum:", np.nanmax(img))
    print("Aantal NaN-waarden:", np.isnan(img).sum())

    # Vier spectrogrammen visualiseren
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for k, ax in enumerate(axes.flat):
        im = ax.imshow(
            img[:, :, k],
            aspect="auto",
            origin="lower"
        )
        ax.set_title(NAMES[k])
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Frequentieband")
        fig.colorbar(im, ax=ax)

    plt.suptitle(f"EEG-spectrogrammen: {test_file.stem}")
    plt.tight_layout()
    plt.show()
