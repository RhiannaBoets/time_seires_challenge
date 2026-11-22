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
import argparse #instellingen meegeven via terminal
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed    #parallel processing
from scipy.linalg import eigh  #eigenvalue decomposition, gebruikt om de eigenwaarden en eigenvectoren van een matrix te berekenen. In dit geval wordt het gebruikt om de generalised eigenvalue problem op te lossen bij het trainen van de Weighted Fisher's Discriminant Analysis (FDA) voor soft multi-class labels.    
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier #meerdere eenvoudige modellen (meestal beslisbomen) na elkaar opbouwt om nauwkeurige voorspellingen te maken voor regressie en classificatie
from sklearn.linear_model import LogisticRegression #klassieke lineaire classificatie, waarbij de kans op een bepaalde klasse wordt gemodelleerd als een logistische functie van de inputkenmerken
from sklearn.model_selection import GroupKFold  #verdeelt de dataset in K-folds, waarbij de gegevens worden gegroepeerd op basis van een bepaalde kolom (bijvoorbeeld patient_id) om te voorkomen dat gegevens van dezelfde groep in zowel de trainings- als validatiesets terechtkomen
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    df = df.reset_index()   # soft labels, een kans in plaats van een harde label (0 of 1) voor elke EEG

    # Quick mode: keep only the EEGs of a random subset of spectrograms. Een kleinere data set selecteren voor snellere tests. De random_state zorgt ervoor dat dezelfde subset wordt geselecteerd bij herhaalde runs.
    if max_specs:
        keep = pd.Series(df.spectrogram_id.unique()).sample(n=min(max_specs, df.spectrogram_id.nunique()),
                                                             random_state=seed)
        df = df[df.spectrogram_id.isin(keep)].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- features
def band_columns(columns):  #dit zijn de power kolommen van de spectrogrammen, die de frequentiebanden representeren
    """Return {(region, band): [column indices]} for the 400 spectrogram columns like 'LL_1.17'."""
    out = {}
    for r in REGIONS:
        for b, (lo, hi) in BANDS.items():
            out[(r, b)] = [i for i, c in enumerate(columns)
                           if c.startswith(r + "_") and lo <= float(c.split("_")[1]) < hi]
    return out  #voor vier regio's en vier frequentiebanden worden de kolomindices van de spectrogrammen teruggegeven


def window_features(spec, cols): #hier worden de 44 kolommen van de spectrogrammen samengevoegd tot 88 features per EEG. De features zijn gebaseerd op de logaritmische band power, de relatieve band power en de asymmetrie tussen de linker- en rechterhersenhelft.
    """Compute the 88 features for one 10-minute window.

    spec: float array of shape (time steps, 400 power columns).
    cols: output of band_columns().
    """
    feats = {}
    total = {r: 0.0 for r in REGIONS}   # total power per region, for relative band power
    band_power = {}
    for (r, b), idx in cols.items():
        p = np.nanmean(spec[:, idx], axis=1)                     # mean power in the band, per time step
        lp = np.log1p(np.nan_to_num(p, nan=0.0).clip(min=0))     # log scale; missing values -> 0, vervangt ontbrekende waarden door 0, voorkomt negatieve powerwaarden en voert logarithmische transformatie uit
                                                                 # logtransformatie maakt grote verschillen in powerwaarden beter hanteerbaar en kan helpen bij het identificeren van patronen in de data
        feats[f"{r}_{b}_mean"] = lp.mean()  #gemiddelde activiteit in de frequentieband over de tijd
        feats[f"{r}_{b}_std"] = lp.std()    #hoeveel de activiteit in de frequentieband varieert over de tijd
        feats[f"{r}_{b}_min"] = lp.min()    #laagst gemeten activiteit in de frequentieband over de tijd
        feats[f"{r}_{b}_max"] = lp.max()    #hoogst gemeten activiteit in de frequentieband over de tijd
        band_power[(r, b)] = np.nanmean(p) if np.isfinite(p).any() else 0.0
        total[r] += band_power[(r, b)]
    for (r, b), v in band_power.items():                         # share of the region's total power
        feats[f"{r}_{b}_rel"] = v / (total[r] + 1e-6)           #relatieve band power: de verhouding van de power in een specifieke frequentieband tot de totale power in die regio, met een kleine constante toegevoegd om deling door nul te voorkomen
    for b in BANDS:                                              # left minus right hemisphere, vergelijkt de activiteit in de linker- en rechterhersenhelft voor elke frequentieband
        feats[f"asym_lat_{b}"] = feats[f"LL_{b}_mean"] - feats[f"RL_{b}_mean"]
        feats[f"asym_par_{b}"] = feats[f"LP_{b}_mean"] - feats[f"RP_{b}_mean"]
    return feats

#verliezen we informatie door het spectogram te comprimeren tot 88 features? Ja, maar het is een trade-off tussen informatierijkdom en modelcomplexiteit. Het gebruik van samengevoegde features kan helpen bij het verminderen van ruis en overfitting, terwijl het nog steeds belangrijke patronen in de data behoudt.

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
#Wat is het verschil tussen de training en test spectrogrammen? Training spectrogrammen bevatten gegevens over het hele EEG-signaal, terwijl test spectrogrammen korter zijn en zich richten op een specifiek segment van het signaal. Dit verschil beïnvloedt hoe de functies worden berekend, met name de selectie van het tijdvenster voor feature-extractie.


def build_features(meta, spec_dir, is_test=False, n_jobs=-1): #n_jobs=-1 betekent dat alle beschikbare CPU-kernen worden gebruikt voor parallelle verwerking. Dit kan de verwerkingstijd aanzienlijk verkorten, vooral bij grote datasets.
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
def kl_score(y_true, y_pred, eps=1e-15):    #meet hoe verschillend de voorspelde kansverdeling is van de werkelijke verdeling. Een lagere KL-score duidt op een betere voorspelling.
    """Mean KL divergence KL(y_true || y_pred) over samples, as in the competition metric."""
    p = np.clip(y_pred, eps, 1)
    p = p / p.sum(axis=1, keepdims=True)
    t = y_true
    return np.mean(np.sum(np.where(t > 0, t * np.log(np.where(t > 0, t, 1) / p), 0), axis=1))


def expand(X, Y): #scikit learn classifiers typically expect hard labels (one class per sample), but in this case, the labels are soft (probability distributions over classes). The expand function transforms the soft labels into a format suitable for training by repeating each sample for each class and assigning weights based on the soft label probabilities.
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
#gewicht bepaalt hoeveel invloed een bepaald voorbeeld heeft op het leren van het model. Voorbeelden met hogere gewichten dragen meer bij aan de parameterupdates tijdens het trainen, terwijl voorbeelden met lagere gewichten minder invloed hebben. Dit is vooral belangrijk bij soft labels, omdat het model moet leren van de onzekerheid in de labels en niet alleen van harde beslissingen.

class SoftLabelKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        self.scaler = StandardScaler()  #features standaardiseren door het gemiddelde en de standaarddeviatie van de trainingsgegevens te gebruiken. Dit helpt om de invloed van verschillende schalen van de features te verminderen en kan de prestaties van het KNN-model verbeteren.
        X_scaled = self.scaler.fit_transform(X)

        self.Y = Y.copy()   #soft labels worden opgeslagen voor gebruik bij het voorspellen van de waarschijnlijkheden van de klassen voor nieuwe voorbeelden. De Y-matrix bevat de kansverdelingen over de klassen voor elk trainingsvoorbeeld, en deze informatie wordt gebruikt om de voorspellingen van het KNN-model te berekenen op basis van de buren van een nieuw voorbeeld.

        self.neighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="euclidean"
        )
        self.neighbors.fit(X_scaled)

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)

        distances, indices = self.neighbors.kneighbors(X_scaled)

        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        probabilities = np.sum(
            self.Y[indices] * weights[:, :, None],
            axis=1
        )

        return probabilities


def make_model(name, seed=42):
    """The two candidate models. Logistic regression needs standardised features; gradient boosting does not."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)) #model leert een gewogen som van de inputkenmerken en past een logistische functie toe om de kans op een bepaalde klasse te voorspellen. C bepaalt de sterkte van de regularisatie, waarbij een lagere waarde meer regularisatie betekent en overfitting kan helpen voorkomen
    
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights='uniform') #KNN-classifier maakt voorspellingen op basis van de meest voorkomende klasse onder de k dichtstbijzijnde trainingsvoorbeelden. De parameter n_neighbors bepaalt het aantal buren dat wordt overwogen bij het maken van een voorspelling, terwijl de parameter weights bepaalt hoe de invloed van de buren wordt gewogen (bijvoorbeeld op basis van afstand).
    
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed) #bouwt reeks van beslisbomen op, waarbij elke boom wordt getraind om de fouten van de vorige bomen te corrigeren. Het model combineert de voorspellingen van alle bomen om een uiteindelijke voorspelling te maken. De hyperparameters zoals learning_rate, max_iter, max_leaf_nodes en l2_regularization beïnvloeden de complexiteit en prestaties van het model.


MODEL_NAMES = ["logreg", "KNN","hgb"]


def fit(name, X, Y, seed=42): #model trainen 
    """Fit model `name` on features X and soft labels Y (rows are vote distributions)."""
    if name == "KNN":
        return SoftLabelKNN(n_neighbors=5).fit(X, Y)
    
    Xe, ye, we = expand(X, Y)
    m = make_model(name, seed)  #soft labels uitgebreid naar gewogen observaties
    if name == "logreg":   # in a Pipeline the weight must be routed to the final step
        m.fit(Xe, ye, logisticregression__sample_weight=we) #model leert relatie tussen de 88 features en de zes klassen, waarbij de gewichten aangeven hoe belangrijk elk voorbeeld is bij het leren van de parameters van het model
    else:
        m.fit(Xe, ye, sample_weight=we)
    return m


def to_matrix(frame, feats):
    """Feature table -> float32 array with missing / infinite values replaced by 0."""
    return np.nan_to_num(frame[feats].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def cross_validate(X, Y, groups, num_votes, n_folds, seed):
    """Patient-grouped K-fold CV for both models; prints scores and returns the better model's name."""
    prior = np.tile(Y.mean(axis=0), (len(Y), 1))   # trivial reference: always predict the mean distribution. prior baseline berekend op basis van de gemiddelde verdeling van de klassen in de trainingsset. Het dient als een eenvoudige referentie om te vergelijken hoe goed de getrainde modellen presteren ten opzichte van een model dat altijd dezelfde voorspelling doet, namelijk de gemiddelde verdeling van de klassen.
    hi = num_votes >= 10                            # EEGs with at least 10 expert votes
    oof = {name: np.zeros_like(Y) for name in MODEL_NAMES}   # out-of-fold predictions per model
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, Y, groups)):  #hoe goed presteert het model op ongeziene data, waarbij de gegevens worden gegroepeerd op basis van patient_id om te voorkomen dat gegevens van dezelfde patiënt in zowel de trainings- als validatiesets terechtkomen
        for name in MODEL_NAMES:
            oof[name][va] = fit(name, X[tr], Y[tr], seed).predict_proba(X[va])
        print(f"fold {fold}: " + ", ".join(f"{n}={kl_score(Y[va], oof[n][va]):.4f}" for n in MODEL_NAMES))
    print(f"\nCV KL (all EEGs)        prior={kl_score(Y, prior):.4f} " +
          " ".join(f"{n}={kl_score(Y, oof[n]):.4f}" for n in MODEL_NAMES))
    print(f"CV KL (>=10 votes only) prior={kl_score(Y[hi], prior[hi]):.4f} " +
          " ".join(f"{n}={kl_score(Y[hi], oof[n][hi]):.4f}" for n in MODEL_NAMES))
    return min(MODEL_NAMES, key=lambda n: kl_score(Y, oof[n]))   # lower KL is better


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

    # 3. Cross-validation and model choice
    if args.no_cv:
        best = "hgb"
    else:
        best = cross_validate(X, Y, df["patient_id"].to_numpy(), df["num_votes"].to_numpy(),
                              args.folds, args.seed)    #model met de laagste KL-score wordt gekozen als het beste model
    print("model for submission:", best)

    # 4. Refit on all training data and predict the test set
    final = fit(best, X, Y, args.seed)  #gekozen model wordt opnieuw getraind op alle trainingsgegevens om de uiteindelijke voorspellingen te maken. nu de volledige trianingsset dus ook dubbele patienten en EEGs met minder dan 10 stemmen worden gebruikt om het model te trainen, zodat het kan profiteren van alle beschikbare informatie en mogelijk betere voorspellingen kan doen op de testset.
    test = pd.read_csv(args.data_dir / "test.csv")
    T_df = build_features(test, args.data_dir / "test_spectrograms", is_test=True, n_jobs=args.n_jobs)
    T_df = test[["eeg_id"]].merge(T_df, on="eeg_id", how="left")   # keep the order of test.csv

    # Submission format: eeg_id plus one predicted probability per class (each row sums to 1).
    sub = pd.DataFrame({"eeg_id": T_df["eeg_id"]})
    sub[CLASSES] = final.predict_proba(to_matrix(T_df, feats))
    sub.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(sub)} rows")


if __name__ == "__main__":   # required for parallel workers on macOS and Windows
    main()
