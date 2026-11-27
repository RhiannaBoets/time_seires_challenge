
import pandas as pd

# Bestanden inlezen
train = pd.read_csv("data/hms/train.csv")
eeg_features = pd.read_csv("eeg_features_train.csv")

# Aantal unieke EEG-ID's
print("Train EEG-ID's:", train["eeg_id"].nunique())
print("Feature EEG-ID's:", eeg_features["eeg_id"].nunique())

# Controleren welke EEG-ID's ontbreken
train_ids = set(train["eeg_id"].unique())
feature_ids = set(eeg_features["eeg_id"].unique())

print("EEG-ID's in train zonder features:", len(train_ids - feature_ids))
print("EEG-ID's met features maar niet in train:", len(feature_ids - train_ids))

# Controleren op dubbele ID's
print("Dubbele feature-ID's:", eeg_features["eeg_id"].duplicated().sum())

# Merge uitvoeren
merged = train[["eeg_id"]].drop_duplicates().merge(
    eeg_features,
    on="eeg_id",
    how="left",
    validate="one_to_one"
)

print("Merged shape:", merged.shape)
print("Ontbrekende featurewaarden:", merged.isna().sum().sum())
