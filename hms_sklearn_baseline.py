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
from sklearn.ensemble import HistGradientBoostingClassifier #meerdere eenvoudige modellen (meestal beslisbomen) na elkaar opbouwt om nauwkeurige voorspellingen te maken voor regressie en classificatie
from sklearn.linear_model import LogisticRegression #klassieke lineaire classificatie, waarbij de kans op een bepaalde klasse wordt gemodelleerd als een logistische functie van de inputkenmerken
from sklearn.model_selection import GroupKFold  #verdeelt de dataset in K-folds, waarbij de gegevens worden gegroepeerd op basis van een bepaalde kolom (bijvoorbeeld patient_id) om te voorkomen dat gegevens van dezelfde groep in zowel de trainings- als validatiesets terechtkomen
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


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

def make_model(name, seed=42):
    """The two candidate models. Logistic regression needs standardised features; gradient boosting does not."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)) #model leert een gewogen som van de inputkenmerken en past een logistische functie toe om de kans op een bepaalde klasse te voorspellen. C bepaalt de sterkte van de regularisatie, waarbij een lagere waarde meer regularisatie betekent en overfitting kan helpen voorkomen
    elif name == "catboost":
        return CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5,  # deze waarde gaan we aanpassen
            loss_function="MultiClass",
            verbose=False,
            random_seed=seed,
            thread_count=2
        )
    elif name == "xgboost":
        return XGBClassifier(learning_rate=0.05, n_estimators=300, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed)
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed) #bouwt reeks van beslisbomen op, waarbij elke boom wordt getraind om de fouten van de vorige bomen te corrigeren. Het model combineert de voorspellingen van alle bomen om een uiteindelijke voorspelling te maken. De hyperparameters zoals learning_rate, max_iter, max_leaf_nodes en l2_regularization beïnvloeden de complexiteit en prestaties van het model.


MODEL_NAMES = ["logreg", "hgb", "catboost", "xgboost"]


def fit(name, X, Y, seed=42): #model trainen 
    """Fit model `name` on features X and soft labels Y (rows are vote distributions)."""
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
    """Patient-grouped CV and evaluation of CatBoost/HGB ensembles."""

    prior = np.tile(Y.mean(axis=0), (len(Y), 1))
    hi = num_votes >= 10

    oof = {
        name: np.zeros_like(Y)
        for name in MODEL_NAMES
    }

    for fold, (tr, va) in enumerate(
        GroupKFold(n_splits=n_folds).split(X, Y, groups)
    ):
        for name in MODEL_NAMES:
            model = fit(name, X[tr], Y[tr], seed)
            oof[name][va] = model.predict_proba(X[va])

        print(
            f"fold {fold}: "
            + ", ".join(
                f"{name}={kl_score(Y[va], oof[name][va]):.4f}"
                for name in MODEL_NAMES
            )
        )

    # Individual model results
    print("\nCV KL (all EEGs)")
    for name in MODEL_NAMES:
        print(f"{name}: {kl_score(Y, oof[name]):.4f}")

    print("\nCV KL (>=10 votes only)")
    for name in MODEL_NAMES:
        print(f"{name}: {kl_score(Y[hi], oof[name][hi]):.4f}")

    # Combine CatBoost and HGB predictions.
    cat_pred = oof["catboost"]
    hgb_pred = oof["hgb"]

    weights = [0.0, 0.25, 0.5, 0.7, 0.8, 1.0]

    print("\nEnsemble results (weight = CatBoost)")
    ensemble_results = {}

    for weight in weights:
        pred = weight * cat_pred + (1 - weight) * hgb_pred

        score_all = kl_score(Y, pred)
        score_hi = kl_score(Y[hi], pred[hi])

        ensemble_results[weight] = score_all

        print(
            f"CatBoost {weight:.0%} + HGB {1-weight:.0%}: "
            f"{1-weight:.0%}: "
            f"all={score_all:.4f}, >=10 votes={score_hi:.4f}"
        )

    # Select the best individual model or ensemble.
    individual_scores = {
        name: kl_score(Y, oof[name])
        for name in MODEL_NAMES
    }

    best_name = min(individual_scores, key=individual_scores.get)
    best_score = individual_scores[best_name]
    best_weight = None

    for weight, score in ensemble_results.items():
        if score < best_score:
            best_score = score
            best_name = "ensemble"
            best_weight = weight

    print(f"\nBest model: {best_name}")
    if best_name == "ensemble":
        print(f"Best CatBoost weight: {best_weight:.0%}")

    return best_name, best_weight



# ----------------------------------------------------------------------------- main

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/hms")
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2
    )

    parser.add_argument(
        "--eeg-features",
        type=Path,
        default=Path("eeg_features_train.csv")
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Labels inlezen
    # --------------------------------------------------

    print("\n1. Labels inlezen...")

    meta = load_labels(
        args.data_dir,
        seed=args.seed
    )

    print("Aantal EEG's:", len(meta))
    print("Aantal patiënten:", meta.patient_id.nunique())

    # --------------------------------------------------
    # 2. Officiële spectrogramfeatures berekenen
    # --------------------------------------------------

    print("\n2. Officiële spectrogramfeatures berekenen...")

    official = build_features(
        meta,
        args.data_dir / "train_spectrograms",
        is_test=False,
        n_jobs=args.n_jobs
    )

    official_count = len([
        c for c in official.columns if c != "eeg_id"
    ])

    print("Aantal officiële features:", official_count)

    # --------------------------------------------------
    # 3. Nieuwe EEG-features inlezen
    # --------------------------------------------------

    print("\n3. Nieuwe EEG-features inlezen...")

    eeg = pd.read_csv(args.eeg_features)

    if "eeg_id" not in eeg.columns:
        raise ValueError("eeg_id ontbreekt in EEG-featurebestand")

    if eeg["eeg_id"].duplicated().any():
        raise ValueError("Dubbele EEG-ID's gevonden")

    # ID-types gelijk maken
    meta["eeg_id"] = meta["eeg_id"].astype("int64")
    official["eeg_id"] = official["eeg_id"].astype("int64")
    eeg["eeg_id"] = eeg["eeg_id"].astype("int64")

    eeg_feature_cols = [
        c for c in eeg.columns if c != "eeg_id"
    ]

    print("Aantal nieuwe EEG-features:", len(eeg_feature_cols))

    # Controleren of alle trainings-EEG's aanwezig zijn
    missing_ids = set(meta.eeg_id) - set(eeg.eeg_id)

    print("Ontbrekende EEG-ID's:", len(missing_ids))

    if missing_ids:
        raise ValueError(
            "Niet alle trainings-EEG's hebben EEG-features."
        )

    # --------------------------------------------------
    # 4. Featuretabellen samenvoegen
    # --------------------------------------------------

    print("\n4. Featuretabellen samenvoegen...")

    frame = meta.merge(
        official,
        on="eeg_id",
        how="left",
        validate="one_to_one"
    )

    frame = frame.merge(
        eeg[["eeg_id"] + eeg_feature_cols],
        on="eeg_id",
        how="left",
        validate="one_to_one"
    )

    official_cols = [
        c for c in official.columns if c != "eeg_id"
    ]

    feature_cols = official_cols + eeg_feature_cols

    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Dubbele featurekolommen gevonden")

    missing_values = frame[feature_cols].isna().sum().sum()

    print("Officiële features:", len(official_cols))
    print("Nieuwe EEG-features:", len(eeg_feature_cols))
    print("Totaal features:", len(feature_cols))
    print("Gecombineerde tabel:", frame.shape)
    print("Ontbrekende waarden:", missing_values)

    if missing_values:
        raise ValueError(
            "Ontbrekende waarden gevonden. Training afgebroken."
        )

    # --------------------------------------------------
    # 5. Modeldata voorbereiden
    # --------------------------------------------------

    print("\n5. Modeldata voorbereiden...")

    # Alle 376 features
    X = to_matrix(frame, feature_cols)

    # Alleen de oorspronkelijke 88 officiële features
    X_official = to_matrix(frame, official_cols)

    Y = frame[CLASSES].to_numpy(dtype=np.float32)

    groups = frame["patient_id"].to_numpy()
    num_votes = frame["num_votes"].to_numpy()

    high_vote = num_votes >= 10

    print("X shape (376 features):", X.shape)
    print("X_official shape (88 features):", X_official.shape)
    print("Y shape:", Y.shape)

    # --------------------------------------------------
    # 6. 5-fold GroupKFold
    # --------------------------------------------------

    print("\n6. Cross-validation starten...")

    # OOF-voorspellingen met alle 376 features
    oof_cat = np.zeros_like(Y, dtype=np.float64)
    oof_hgb = np.zeros_like(Y, dtype=np.float64)

    # OOF-voorspellingen met alleen de oorspronkelijke 88 features
    oof_cat_base = np.zeros_like(Y, dtype=np.float64)
    oof_hgb_base = np.zeros_like(Y, dtype=np.float64)

    splitter = GroupKFold(n_splits=args.folds)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, Y, groups),
        start=1
    ):

        print(f"\nFold {fold}/{args.folds}")

        # CatBoost
        print("CatBoost trainen...")

        cat = fit(
            "catboost",
            X[train_idx],
            Y[train_idx],
            args.seed
        )

        oof_cat[valid_idx] = cat.predict_proba(
            X[valid_idx]
        )

        # HistGradientBoosting
        print("HGB trainen...")

        hgb = fit(
            "hgb",
            X[train_idx],
            Y[train_idx],
            args.seed
        )

        oof_hgb[valid_idx] = hgb.predict_proba(
            X[valid_idx]
        )
        # Baseline: alleen de oorspronkelijke 88 features
        print("Baseline CatBoost (88 features) trainen...")

        cat_base = fit(
            "catboost",
            X_official[train_idx],
            Y[train_idx],
            args.seed
        )

        oof_cat_base[valid_idx] = cat_base.predict_proba(
            X_official[valid_idx]
        )

        print("Baseline HGB (88 features) trainen...")

        hgb_base = fit(
            "hgb",
            X_official[train_idx],
            Y[train_idx],
            args.seed
        )

        oof_hgb_base[valid_idx] = hgb_base.predict_proba(
            X_official[valid_idx]
        )
        print(
            "Fold KL | "
            f"CatBoost: {kl_score(Y[valid_idx], oof_cat[valid_idx]):.4f} | "
            f"HGB: {kl_score(Y[valid_idx], oof_hgb[valid_idx]):.4f}"
        )

    # --------------------------------------------------
    # 7. Individuele resultaten
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("RESULTATEN GECOMBINEERDE FEATURES")
    print("=" * 50)

    print(
        f"CatBoost all: {kl_score(Y, oof_cat):.4f}"
    )

    print(
        f"HGB all:      {kl_score(Y, oof_hgb):.4f}"
    )

    print(
        f"CatBoost >=10 votes: "
        f"{kl_score(Y[high_vote], oof_cat[high_vote]):.4f}"
    )

    print(
        f"HGB >=10 votes: "
        f"{kl_score(Y[high_vote], oof_hgb[high_vote]):.4f}"
    )

    # --------------------------------------------------
    # 8. Ensemble testen
    # --------------------------------------------------

    print("\nENSEMBLE RESULTATEN")

    scores = {}

    for weight in [0.0, 0.25, 0.5, 0.7, 0.8, 1.0]:

        pred = (
            weight * oof_cat
            + (1.0 - weight) * oof_hgb
        )

        score_all = kl_score(Y, pred)

        score_high = kl_score(
            Y[high_vote],
            pred[high_vote]
        )

        scores[weight] = score_all

        print(
            f"CatBoost {weight:.0%} + "
            f"HGB {1-weight:.0%}: "
            f"all={score_all:.4f}, "
            f">=10 votes={score_high:.4f}"
        )

    best_weight = min(scores, key=scores.get)

    print("\n" + "=" * 50)
    print("BESTE ENSEMBLE")
    print("=" * 50)

    print(
        f"CatBoost {best_weight:.0%} + "
        f"HGB {1-best_weight:.0%}"
    )

    print(
        f"KL-score: {scores[best_weight]:.4f}"
    )
    # --------------------------------------------------
    # 9. Eerlijke vergelijking: 88 versus 376 features
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("VERGELIJKING: 88 VERSUS 376 FEATURES")
    print("=" * 55)

    baseline_ensemble = (
        0.70 * oof_cat_base
        + 0.30 * oof_hgb_base
    )

    combined_ensemble = (
        0.70 * oof_cat
        + 0.30 * oof_hgb
    )

    baseline_score = kl_score(Y, baseline_ensemble)
    combined_score = kl_score(Y, combined_ensemble)

    print(f"Baseline (88 features):  {baseline_score:.4f}")
    print(f"Nieuw (376 features):    {combined_score:.4f}")
    print(f"Verbetering:              {baseline_score - combined_score:.4f}")


    # --------------------------------------------------
    # 10. Testfeatures berekenen en samenvoegen
    # --------------------------------------------------

    print("\n10. Testfeatures berekenen...")

    test = pd.read_csv(args.data_dir / "test.csv")

    test_meta = (
        test[["eeg_id", "spectrogram_id"]]
        .drop_duplicates("eeg_id")
        .reset_index(drop=True)
    )

    test_official = build_features(
        test_meta,
        args.data_dir / "test_spectrograms",
        is_test=True,
        n_jobs=args.n_jobs
    )

    eeg_test = pd.read_csv("eeg_features_test.csv")

    test_meta["eeg_id"] = test_meta["eeg_id"].astype("int64")
    test_official["eeg_id"] = test_official["eeg_id"].astype("int64")
    eeg_test["eeg_id"] = eeg_test["eeg_id"].astype("int64")

    if eeg_test["eeg_id"].duplicated().any():
        raise ValueError("Dubbele EEG-ID's in testfeatures.")

    missing_test_ids = set(test_meta.eeg_id) - set(eeg_test.eeg_id)

    if missing_test_ids:
        raise ValueError(
            f"{len(missing_test_ids)} test-EEG's missen EEG-features."
        )

    test_frame = test_meta.merge(
        test_official,
        on="eeg_id",
        how="left",
        validate="one_to_one"
    )

    test_frame = test_frame.merge(
        eeg_test,
        on="eeg_id",
        how="left",
        validate="one_to_one"
    )

    test_feature_cols = official_cols + eeg_feature_cols

    if list(test_feature_cols) != list(feature_cols):
        raise ValueError("Train- en testfeaturekolommen komen niet overeen.")

    if test_frame[test_feature_cols].isna().sum().sum() > 0:
        raise ValueError("Ontbrekende waarden in testfeatures.")

    X_test = to_matrix(test_frame, feature_cols)

    print("Testfeatures:", X_test.shape)

    # --------------------------------------------------
    # 11. Modellen trainen op alle trainingsdata
    # --------------------------------------------------

    print("\n11. Definitieve modellen trainen...")

    final_cat = fit(
        "catboost",
        X,
        Y,
        args.seed
    )

    final_hgb = fit(
        "hgb",
        X,
        Y,
        args.seed
    )

    # --------------------------------------------------
    # 12. Voorspellingen combineren
    # --------------------------------------------------

    print("\n12. Testvoorspellingen maken...")

    pred_cat = final_cat.predict_proba(X_test)
    pred_hgb = final_hgb.predict_proba(X_test)

    predictions = (
        0.70 * pred_cat
        + 0.30 * pred_hgb
    )

    predictions = predictions / predictions.sum(
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------
    # 13. Submission opslaan
    # --------------------------------------------------

    submission = pd.DataFrame(
        predictions,
        columns=CLASSES
    )

    submission.insert(
        0,
        "eeg_id",
        test_frame["eeg_id"].values
    )

    submission.to_csv("submission_eeg_ensemble.csv", index=False)

    print("\nSubmission opgeslagen:")
    print("submission_eeg_ensemble.csv")
    print("Vorm:", submission.shape)
    print(submission.head())

    print("\nExperiment en submission afgerond.")


    print(
        "\nLet op: de beste ensembleweging is geselecteerd "
        "op dezelfde OOF-voorspellingen. De score is dus "
        "een modelselectiescore, geen onafhankelijke testscore."
    )


if __name__ == "__main__":
    main()
