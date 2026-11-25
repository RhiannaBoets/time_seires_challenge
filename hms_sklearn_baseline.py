"""HMS – Harmful Brain Activity Classification: a simple scikit-learn baseline (local script).

Same method as hms_sklearn_baseline.ipynb, packaged as a command-line script for running on
your own computer:

1. Labels: expert votes per EEG are normalised to a probability distribution over six classes.
2. Features: 88 per EEG from the official spectrograms (10-minute window around the labelled
   segment): log band power (delta/theta/alpha/beta) per scalp region summarised over time,
   relative band power and left-right asymmetry.
3. Models: LogisticRegression and HistGradientBoostingClassifier, trained on soft labels by
   repeating each EEG once per class with sample_weight = vote fraction.
4. Validation: 5-fold GroupKFold by patient_id; the model with the better CV score is used.
5. Submission: refit on all training data, predict test.csv, write submission.csv.

Usage:
    python hms_sklearn_baseline.py --data-dir /path/to/hms-harmful-brain-activity-classification
    python hms_sklearn_baseline.py --data-dir ... --mode quick      # random 3,000 spectrograms
    python hms_sklearn_baseline.py --data-dir ... --no-cv           # skip cross-validation

Requires: numpy, pandas, pyarrow, scikit-learn, joblib.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# Some spectrograms contain frequency bands that are entirely NaN; numpy warns about the
# resulting empty means. These cases are handled explicitly below, so the warnings are muted.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Vote columns in train.csv, in the order required by the submission file.
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
# The official spectrograms average the bipolar EEG channels into four scalp regions:
# LL = left lateral, RL = right lateral, LP = left parasagittal, RP = right parasagittal.
REGIONS = ["LL", "RL", "LP", "RP"]
# Classical EEG frequency bands in Hz (the spectrograms cover roughly 0.5-20 Hz).
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 20)}


# ----------------------------------------------------------------------------- labels
def load_labels(data_dir, max_specs=None, seed=42):
    """Return one row per EEG with its spectrogram id, label offsets, patient id and vote distribution."""
    train = pd.read_csv(data_dir / "train.csv")

    # train.csv has several labelled sub-segments per eeg_id: sum their votes and keep the offset range.
    agg = {"spectrogram_id": "first", "spectrogram_label_offset_seconds": ["min", "max"], "patient_id": "first"}
    agg.update({c: "sum" for c in CLASSES})
    df = train.groupby("eeg_id").agg(agg)
    df.columns = ["spectrogram_id", "off_min", "off_max", "patient_id"] + CLASSES

    # Convert vote counts into a probability distribution (each row sums to 1).
    df["num_votes"] = df[CLASSES].sum(axis=1)
    df[CLASSES] = df[CLASSES].values / df["num_votes"].values[:, None]
    df = df.reset_index()

    # Quick mode: keep only the EEGs of a random subset of spectrograms.
    if max_specs:
        keep = pd.Series(df.spectrogram_id.unique()).sample(n=min(max_specs, df.spectrogram_id.nunique()),
                                                             random_state=seed)
        df = df[df.spectrogram_id.isin(keep)].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- features
def band_columns(columns):
    """Return {(region, band): [column indices]} for the 400 spectrogram columns like 'LL_1.17'."""
    out = {}
    for r in REGIONS:
        for b, (lo, hi) in BANDS.items():
            out[(r, b)] = [i for i, c in enumerate(columns)
                           if c.startswith(r + "_") and lo <= float(c.split("_")[1]) < hi]
    return out


def window_features(spec, cols):
    """Compute the 88 features for one 10-minute window.

    spec: float array of shape (time steps, 400 power columns).
    cols: output of band_columns().
    """
    feats = {}
    total = {r: 0.0 for r in REGIONS}   # total power per region, for relative band power
    band_power = {}
    for (r, b), idx in cols.items():
        p = np.nanmean(spec[:, idx], axis=1)                     # mean power in the band, per time step
        lp = np.log1p(np.nan_to_num(p, nan=0.0).clip(min=0))     # log scale; missing values -> 0
        feats[f"{r}_{b}_mean"] = lp.mean()
        feats[f"{r}_{b}_std"] = lp.std()
        feats[f"{r}_{b}_min"] = lp.min()
        feats[f"{r}_{b}_max"] = lp.max()
        band_power[(r, b)] = np.nanmean(p) if np.isfinite(p).any() else 0.0
        total[r] += band_power[(r, b)]
    for (r, b), v in band_power.items():                         # share of the region's total power
        feats[f"{r}_{b}_rel"] = v / (total[r] + 1e-6)
    for b in BANDS:                                              # left minus right hemisphere
        feats[f"asym_lat_{b}"] = feats[f"LL_{b}_mean"] - feats[f"RL_{b}_mean"]
        feats[f"asym_par_{b}"] = feats[f"LP_{b}_mean"] - feats[f"RP_{b}_mean"]
    return feats


def spectrogram_features(path, windows, is_test):
    """Read one spectrogram file and compute features for every EEG that uses it.

    windows: list of (eeg_id, off_min, off_max); offsets are the label positions in seconds.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # also mute warnings in worker processes
    spec = pd.read_parquet(path)
    cols = band_columns(list(spec.columns[1:]))
    spec = spec.iloc[:, 1:].to_numpy(dtype=np.float32)          # drop the 'time' column
    rows = []
    for eeg_id, off_min, off_max in windows:
        if is_test:
            # Test spectrograms are short and centred on the segment: take their middle 10 minutes.
            start = max(0, (len(spec) - 300) // 2)
        else:
            # Training spectrograms cover the whole recording. Start at the midpoint of the labelled
            # offsets; offsets are in seconds and rows are 2 s apart, hence the division by 4 (= /2 /2).
            start = int((off_min + off_max) // 4)
        feats = window_features(spec[start:start + 300], cols)
        feats["eeg_id"] = eeg_id
        rows.append(feats)
    return rows


def build_features(meta, spec_dir, is_test=False, n_jobs=-1):
    """Compute features for all EEGs in `meta`, reading each spectrogram file once (in parallel)."""
    t0 = time.time()
    if is_test:   # test.csv has no label offsets
        meta = meta.assign(off_min=0.0, off_max=0.0)
    jobs = [delayed(spectrogram_features)(spec_dir / f"{sid}.parquet",
                                          list(zip(g.eeg_id, g.off_min, g.off_max)), is_test)
            for sid, g in meta.groupby("spectrogram_id")]
    results = Parallel(n_jobs=n_jobs, batch_size=32)(jobs)
    out = pd.DataFrame([row for rows in results for row in rows])
    print(f"{len(jobs)} spectrograms -> {len(out)} feature rows in {time.time() - t0:.0f}s")
    return out


# ----------------------------------------------------------------------------- models
def kl_score(y_true, y_pred, eps=1e-15):
    """Mean KL divergence KL(y_true || y_pred) over samples, as in the competition metric."""
    p = np.clip(y_pred, eps, 1)
    p = p / p.sum(axis=1, keepdims=True)
    t = y_true
    return np.mean(np.sum(np.where(t > 0, t * np.log(np.where(t > 0, t, 1) / p), 0), axis=1))


def expand(X, Y):
    """Turn soft labels into weighted hard labels.

    Each sample is repeated once per class (label = class index, weight = vote fraction);
    copies with zero weight are dropped.
    """
    n, k = Y.shape
    Xr = np.repeat(X, k, axis=0)
    yr = np.tile(np.arange(k), n)
    wr = Y.reshape(-1)
    keep = wr > 0
    return Xr[keep], yr[keep], wr[keep]


class SpectrogramData:
    """Verwijzingen naar spectrogrambestanden, zonder alles in RAM te laden."""
    def __init__(self, meta, spec_dir):
        self.meta = meta.reset_index(drop=True).copy()
        self.spec_dir = Path(spec_dir)

    def __len__(self):
        return len(self.meta)

    def subset(self, indices):
        return SpectrogramData(
            self.meta.iloc[indices].reset_index(drop=True),
            self.spec_dir
        )


class HMSSpectrogramDataset(Dataset):
    """Leest per EEG het juiste venster uit het bijbehorende spectrogram."""

    def __init__(self, data, labels=None, cache_size=8):
        self.data = data
        self.labels = labels
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.data)

    
    def load_spectrogram(self, spec_id):
    # Convert numeric spectrogram IDs to integer strings.
        if isinstance(spec_id, (float, np.floating)) and spec_id.is_integer():
            spec_id = str(int(spec_id))
        else:
            spec_id = str(spec_id)

        if spec_id in self.cache:
            self.cache.move_to_end(spec_id)
            return self.cache[spec_id]

        path = self.data.spec_dir / f"{spec_id}.parquet"

        spec = pd.read_parquet(path)
        values = spec.iloc[:, 1:].to_numpy(dtype=np.float32)

        self.cache[spec_id] = values

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return values


    def __getitem__(self, idx):
        row = self.data.meta.iloc[idx]
        spec = self.load_spectrogram(row["spectrogram_id"])

        # Trainingsdata: selecteer het venster rond de label-offsets.
        if "off_min" in row.index and "off_max" in row.index:
            start = int((row["off_min"] + row["off_max"]) // 4)
        else:
            # Testdata: neem het middelste venster.
            start = max(0, (len(spec) - 300) // 2)

        start = min(start, max(0, len(spec) - 300))
        window = spec[start:start + 300]

        # Maak elk venster exact 300 tijdstappen lang.
        if len(window) < 300:
            padding = np.zeros(
                (300 - len(window), spec.shape[1]),
                dtype=np.float32
            )
            window = np.concatenate([window, padding], axis=0)

        # Logtransformatie en behandeling van ontbrekende waarden.
        window = np.nan_to_num(
            window, nan=0.0, posinf=0.0, neginf=0.0
        )
        window = np.log1p(np.maximum(window, 0.0))

        # Normalisatie per spectrogramsegment.
        mean = window.mean()
        std = window.std()
        window = (window - mean) / (std + 1e-6)

        # CNN verwacht [channels, height, width].
        x = torch.from_numpy(
            np.ascontiguousarray(window)
        ).float().unsqueeze(0)

        if self.labels is None:
            return x

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class HMSConvNet(nn.Module):
    """2D convolutional neural network voor EEG-spectrogrammen."""

    def __init__(self, n_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class SoftLabelCNN:
    """CNN met dezelfde fit() en predict_proba() interface als sklearn."""

    def __init__(self, epochs=4, batch_size=16, lr=0.001, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.model = None
        self.device = torch.device("cpu")

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        dataset = HMSSpectrogramDataset(X, labels=Y)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.model = HMSConvNet(n_classes=Y.shape[1]).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_samples = 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                logits = self.model(xb)
                log_probs = F.log_softmax(logits, dim=1)

                loss = F.kl_div(
                    log_probs,
                    yb,
                    reduction="batchmean"
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(xb)
                total_samples += len(xb)

            print(
                f"    CNN epoch {epoch + 1}/{self.epochs}: "
                f"KL={total_loss / total_samples:.4f}"
            )

        return self

    def predict_proba(self, X):
        dataset = HMSSpectrogramDataset(X, labels=None)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                probabilities = F.softmax(self.model(xb), dim=1)
                predictions.append(probabilities.cpu().numpy())

        return np.concatenate(predictions, axis=0)




def make_model(name, seed=42):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000)
        )

    elif name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed
        )

    elif name == "cnn":
        return SoftLabelCNN(
            epochs=4,
            batch_size=16,
            lr=0.001,
            seed=seed
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["logreg", "hgb", "cnn"]



def fit(name, X, Y, seed=42):
    model = make_model(name, seed)

    if name == "cnn":
        model.fit(X, Y)

    else:
        X_expanded, y_expanded, weights = expand(X, Y)

        if name == "logreg":
            model.fit(
                X_expanded,
                y_expanded,
                logisticregression__sample_weight=weights
            )
        else:
            model.fit(
                X_expanded,
                y_expanded,
                sample_weight=weights
            )

    return model



def to_matrix(frame, feats):
    """Feature table -> float32 array with missing / infinite values replaced by 0."""
    return np.nan_to_num(frame[feats].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def cross_validate(X, X_spectrograms, Y, groups, num_votes,
                   n_folds, seed):

    prior = np.tile(Y.mean(axis=0), (len(Y), 1))
    hi = num_votes >= 10

    oof = {
        name: np.zeros_like(Y)
        for name in MODEL_NAMES
    }

    splitter = GroupKFold(n_splits=n_folds)

    for fold, (tr, va) in enumerate(
        splitter.split(X, Y, groups)
    ):
        print(f"\nfold {fold}")

        for name in MODEL_NAMES:

            if name == "cnn":
                # CNN gebruikt de spectrogrammen.
                X_train = X_spectrograms.subset(tr)
                X_valid = X_spectrograms.subset(va)

            else:
                # LogReg en HGB gebruiken de 88 features.
                X_train = X[tr]
                X_valid = X[va]

            model = fit(name, X_train, Y[tr], seed + fold)

            oof[name][va] = model.predict_proba(X_valid)

        print(
            "  " + ", ".join(
                f"{name}={kl_score(Y[va], oof[name][va]):.4f}"
                for name in MODEL_NAMES
            )
        )

    print(
        f"\nCV KL (all EEGs) prior={kl_score(Y, prior):.4f} "
        + " ".join(
            f"{name}={kl_score(Y, oof[name]):.4f}"
            for name in MODEL_NAMES
        )
    )

    print(
        f"\nCV KL (>=10 votes only) prior={kl_score(Y[hi], prior[hi]):.4f} "
        + " ".join(
            f"{name}={kl_score(Y[hi], oof[name][hi]):.4f}"
            for name in MODEL_NAMES
        )
    )

    return min(
        MODEL_NAMES,
        key=lambda name: kl_score(Y, oof[name])
    )


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="HMS simple scikit-learn baseline")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="folder that contains train.csv, test.csv, train_spectrograms/ and test_spectrograms/")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                        help="full: all training spectrograms; quick: a random 3,000 for a fast check")
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation and use gradient boosting")
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for reading spectrograms (-1 = all)")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="where to write predictions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Labels
    df = load_labels(args.data_dir, max_specs=3000 if args.mode == "quick" else None, seed=args.seed)
    print(df.shape, "EEGs from", df.patient_id.nunique(), "patients,", df.spectrogram_id.nunique(), "spectrograms")

    # 2. Features
    X_df = build_features(df, args.data_dir / "train_spectrograms", n_jobs=args.n_jobs)
    df = df.merge(X_df, on="eeg_id")
    feats = [c for c in X_df.columns if c != "eeg_id"]
    print(len(feats), "features")
    X = to_matrix(df, feats)
    Y = df[CLASSES].to_numpy()
    # Spectrogramverwijzingen voor de CNN.
    # De bestanden worden pas tijdens training en voorspelling ingelezen.
    X_spectrograms = SpectrogramData(
        df,
        args.data_dir / "train_spectrograms"
    )

    # 3. Cross-validation and model choice
    if args.no_cv:
        best = "hgb"
    else:
        best = cross_validate(
                        X,
                        X_spectrograms,
                        Y,
                        df["patient_id"].to_numpy(),
                        df["num_votes"].to_numpy(),
                        args.folds,
                        args.seed
                    )
    print("model for submission:", best)

    # 4. Refit on all training data and predict the test set
    
    # 4. Refit on all training data and predict the test set.
    test = pd.read_csv(args.data_dir / "test.csv")

    if best == "cnn":
        # CNN gebruikt rechtstreeks de test-spectrogrammen.
        X_test = SpectrogramData(
            test,
            args.data_dir / "test_spectrograms"
        )

        final = fit("cnn", X_spectrograms, Y, args.seed)
        predictions = final.predict_proba(X_test)

        sub = pd.DataFrame({"eeg_id": test["eeg_id"]})
        sub[CLASSES] = predictions

    else:
        # Bestaande pipeline voor LogReg en HGB blijft behouden.
        final = fit(best, X, Y, args.seed)

        T_df = build_features(
            test,
            args.data_dir / "test_spectrograms",
            is_test=True,
            n_jobs=args.n_jobs
        )

        T_df = test[["eeg_id"]].merge(
            T_df,
            on="eeg_id",
            how="left"
        )

        sub = pd.DataFrame({"eeg_id": T_df["eeg_id"]})
        sub[CLASSES] = final.predict_proba(
            to_matrix(T_df, feats)
        )

    sub.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(sub)} rows")


    # Submission format: eeg_id plus one predicted probability per class (each row sums to 1).
    sub = pd.DataFrame({"eeg_id": T_df["eeg_id"]})
    sub[CLASSES] = final.predict_proba(to_matrix(T_df, feats))
    sub.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(sub)} rows")


if __name__ == "__main__":   # required for parallel workers on macOS and Windows
    main()
