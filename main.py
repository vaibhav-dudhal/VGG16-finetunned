#!/usr/bin/env python3
"""
Face Spoofing Attack Detection Method for Cyberphysical Space
GB20 - Pimpri Chinchwad College of Engineering, Pune

Implements the method exactly as described in the IEEE paper:

  * Features  : Thepade's Sorted Block Truncation Coding (TSBTC)
                - per colour plane (R, G, B) sort pixel intensities ascending  (Eq. 1)
                - split the sorted array into n contiguous parts               (n_parts)
                - feature = mean intensity of each part                        (Eq. 2)
                - concatenate R, G, B  ->  feature vector of length 3n
  * n_parts   : 2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64   (11 variations)
  * Models    : 19 configurations per n_parts
                3 Tree + 1 intra-family voting ensemble
                4 Boosting + 1 intra-family voting ensemble
                3 Distance & Kernel + 1 intra-family voting ensemble
                3 Probabilistic & Bayes + 1 intra-family voting ensemble
                1 inter-family voting ensemble   (of the 4 family ensembles)
                1 inter-family stacking ensemble (meta-classifier on the 4 family ensembles)
  * Protocol  : stratified 10-fold cross-validation on the NUAA dataset
  * Outputs   : accuracy table (Table III ranking), accuracy-vs-n_parts figure,
                and Precision / Recall / F1 / ROC-AUC / FAR / FRR for the final
                selected configuration (Eqs. 3-8).

Label convention: genuine = 1 (positive), spoof = 0 (negative), so
FAR = FP / (FP + TN) is a spoof accepted as genuine and
FRR = FN / (FN + TP) is a genuine face rejected as spoof.

Dataset layout expected (NUAA "raw" folder):
    raw/ClientRaw/<subject>/*.jpg     -> genuine
    raw/ImposterRaw/<subject>/*.jpg   -> spoof

Usage:
    python main.py                      # full run, as in the paper
    python main.py --max-images 600 --n-values 8,32 --folds 3   # quick smoke test
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
except ImportError as exc:  # the paper uses all tree and boosting models
    sys.exit(
        f"Missing boosting library ({exc}). Run: pip install -r requirements.txt"
    )

SEED = 42
N_PARTS = [2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

FAM_TREE = "Tree"
FAM_BOOST = "Boosting"
FAM_DIST = "Distance & Kernel"
FAM_PROB = "Probabilistic & Bayes"
INTRA = "Intra-Family Voting Ensemble"
INTER_VOTE = "Inter-Family Voting Ensemble"
INTER_STACK = "Inter-Family Stacking Ensemble"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_dataset(data_dir):
    """Return (paths, labels, subject_ids). ClientRaw -> 1 (genuine), ImposterRaw -> 0 (spoof)."""
    if not os.path.exists(data_dir) or not any(os.scandir(data_dir) if os.path.exists(data_dir) else []):
        parent_dir = os.path.dirname(os.path.abspath(data_dir))
        for candidate in [os.path.join(parent_dir, "archive.zip"), os.path.join(data_dir, "archive.zip")]:
            if os.path.isfile(candidate):
                print(f"Dataset folder empty/missing. Auto-extracting from {candidate} ...")
                import zipfile
                with zipfile.ZipFile(candidate, "r") as z:
                    z.extractall(parent_dir)
                break

    paths, labels, subjects = [], [], []
    for root, _, files in os.walk(data_dir):
        parts = [p.lower() for p in os.path.normpath(root).split(os.sep)]
        if "clientraw" in parts:
            label = 1
        elif "imposterraw" in parts:
            label = 0
        else:
            continue
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(root, f))
                labels.append(label)
                subjects.append(os.path.basename(root))
    order = np.argsort(paths)  # deterministic order
    paths = [paths[i] for i in order]
    labels = np.array(labels)[order]
    subjects = np.array(subjects)[order]
    return paths, labels, subjects



# --------------------------------------------------------------------------- #
# TSBTC feature extraction (paper Section III-D, Eq. 1 and 2)
# --------------------------------------------------------------------------- #
def tsbtc_features(image_path, n_values):
    """
    For one image, return {n: feature_vector_of_length_3n} for every n in n_values.
    The sort is done once per colour plane and reused for every n_parts value.
    If M*N is not divisible by n, np.array_split gives parts that differ by at most one pixel.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    planes = rgb.reshape(-1, 3)                 # (M*N, 3): columns are R, G, B
    sorted_planes = np.sort(planes, axis=0)     # Eq. 1: ascending order per plane
    out = {}
    for n in n_values:
        chunks = np.array_split(sorted_planes, n, axis=0)
        means = np.stack([c.mean(axis=0) for c in chunks])   # (n, 3)  Eq. 2
        out[n] = means.T.reshape(-1)                         # [R_1..R_n, G_1..G_n, B_1..B_n]
    return out


def build_feature_sets(paths, n_values, n_jobs):
    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(tsbtc_features)(p, n_values) for p in paths
    )
    valid = np.array([r is not None for r in results])
    feats = {n: np.stack([r[n] for r in results if r is not None]) for n in n_values}
    return feats, valid


# --------------------------------------------------------------------------- #
# Classifiers 
# --------------------------------------------------------------------------- #
def get_families():
    tree = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    }
    boosting = {
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0, iterations=100, thread_count=-1, random_seed=SEED),
    }
    dist = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
        "SVM (RBF Kernel)": SVC(kernel="rbf", cache_size=1000, random_state=SEED),
        "Linear SVM": LinearSVC(dual="auto", max_iter=5000, random_state=SEED),
    }
    prob = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "Bernoulli Naive Bayes": BernoulliNB(),
    }
    return {FAM_TREE: tree, FAM_BOOST: boosting, FAM_DIST: dist, FAM_PROB: prob}


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def make_family_voting(models):
    """Intra-family hard-voting ensemble over every base model of the family."""
    return VotingClassifier(
        estimators=[(_slug(k), clone(m)) for k, m in models.items()], voting="hard"
    )


def make_inter_stacking(families):
    """Meta-classifier trained on the outputs of all family ensembles."""
    estimators = [(_slug(fam), make_family_voting(models)) for fam, models in families.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3,
    )


def majority_vote(pred_matrix):
    """Hard majority vote over rows of a (n_models, n_samples) 0/1 prediction matrix."""
    pred_matrix = np.asarray(pred_matrix)
    return (pred_matrix.sum(axis=0) * 2 > pred_matrix.shape[0]).astype(int)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def make_splitter(n_folds, subject_split):
    if subject_split:
        return StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)


def evaluate_one_n(X, y, groups, n_folds, subject_split):
    """
    Mean cross-validation accuracy for all 19 configurations on one feature set.
    Every base model is fitted once per fold; hard-voting ensembles (intra- and
    inter-family) are computed from those fitted models' predictions, which is
    mathematically identical to refitting a VotingClassifier(voting='hard').
    The stacking ensemble is a real StackingClassifier fitted per fold.
    """
    families = get_families()
    splitter = make_splitter(n_folds, subject_split)
    scores = {}

    def add(key, value):
        scores.setdefault(key, []).append(value)

    for tr, te in splitter.split(X, y, groups if subject_split else None):
        fam_votes = []
        for fam, models in families.items():
            fold_preds = []
            for name, model in models.items():
                m = clone(model).fit(X[tr], y[tr])
                p = m.predict(X[te])
                fold_preds.append(p)
                add((fam, name), accuracy_score(y[te], p))
            vote = majority_vote(fold_preds)
            fam_votes.append(vote)
            add((fam, INTRA), accuracy_score(y[te], vote))

        add(("Ensemble Learning", INTER_VOTE), accuracy_score(y[te], majority_vote(fam_votes)))

        stack = make_inter_stacking(families).fit(X[tr], y[tr])
        add(("Ensemble Learning", INTER_STACK), accuracy_score(y[te], stack.predict(X[te])))

    return {k: float(np.mean(v)) for k, v in scores.items()}


def final_metrics(X, y, groups, n_folds, subject_split):
    """Eqs. 3-8 plus ROC-AUC for the inter-family stacking ensemble, pooled over all folds."""
    families = get_families()
    splitter = make_splitter(n_folds, subject_split)
    y_true, y_pred, y_prob = [], [], []
    for tr, te in splitter.split(X, y, groups if subject_split else None):
        stack = make_inter_stacking(families).fit(X[tr], y[tr])
        y_true.append(y[te])
        y_pred.append(stack.predict(X[te]))
        y_prob.append(stack.predict_proba(X[te])[:, 1])
    y_true, y_pred, y_prob = map(np.concatenate, (y_true, y_pred, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "FAR": fp / (fp + tn) if (fp + tn) else float("nan"),
        "FRR": fn / (fn + tp) if (fn + tp) else float("nan"),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def plot_accuracy(df, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figure.")
        return
    picks = [
        (FAM_TREE, "Decision Tree"),
        (FAM_TREE, INTRA),
        (FAM_BOOST, "XGBoost"),
        (FAM_BOOST, INTRA),
        (FAM_DIST, INTRA),
        (FAM_PROB, INTRA),
        ("Ensemble Learning", INTER_VOTE),
        ("Ensemble Learning", INTER_STACK),
    ]
    plt.figure(figsize=(7, 4.2))
    for col in picks:
        if col in df.columns:
            plt.plot(df.index, df[col] * 100, marker="o", label=col[1])
    plt.xlabel("TSBTC n_parts")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs TSBTC n_parts (10-fold CV)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--folds", type=int, default=10, help="CV folds (paper: 10)")
    ap.add_argument("--n-values", default=",".join(map(str, N_PARTS)), help="comma separated n_parts list")
    ap.add_argument("--max-images", type=int, default=0, help="stratified subsample for quick tests (0 = all images)")
    ap.add_argument("--subject-split", action="store_true",
                    help="keep every subject's images in a single fold (stricter than the paper's protocol)")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for feature extraction")
    ap.add_argument("--no-resume", action="store_true", help="ignore cached per-n results")
    args = ap.parse_args()

    n_values = [int(v) for v in args.n_values.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset from: {args.data_dir}")
    paths, labels, subjects = load_dataset(args.data_dir)
    print(f"Found {len(paths)} images  (genuine={int(labels.sum())}, spoof={int((labels == 0).sum())})")
    if not paths:
        sys.exit("No images found. Expected raw/ClientRaw/<id>/*.jpg and raw/ImposterRaw/<id>/*.jpg")

    if args.max_images and len(paths) > args.max_images:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(paths), args.max_images, replace=False)
        paths = [paths[i] for i in idx]
        labels, subjects = labels[idx], subjects[idx]
        print(f"Quick-test mode: using {len(paths)} images")

    t0 = time.time()
    print(f"Extracting TSBTC features for n_parts = {n_values} ...")
    feats, valid = build_feature_sets(paths, n_values, args.jobs)
    y = labels[valid]
    groups = subjects[valid]
    print(f"Feature extraction done in {time.time() - t0:.1f}s ({int(valid.sum())} usable images)")

    all_acc = {}
    for n in n_values:
        cache = os.path.join(args.out_dir, f"acc_n{n}.json")
        if os.path.exists(cache) and not args.no_resume:
            with open(cache) as fh:
                cached = {tuple(k.split("||")): v for k, v in json.load(fh).items()}
            # Older cache files combined tree and boosting; they cannot provide
            # the separate family results required by this version.
            has_separate_boosting = any(key[0] == FAM_BOOST for key in cached)
            if has_separate_boosting:
                all_acc[n] = cached
                print(f"\n--- n_parts = {n}: loaded cached results ---")
                continue
            print(f"\n--- n_parts = {n}: recomputing for separate tree/boosting families ---")
        print(f"\n--- n_parts = {n}  (feature length {3 * n}) ---")
        t1 = time.time()
        res = evaluate_one_n(feats[n], y, groups, args.folds, args.subject_split)
        for (fam, name), acc in res.items():
            print(f"  {fam:<22} {name:<34} {acc:.4f}")
        print(f"  ({time.time() - t1:.0f}s)")
        all_acc[n] = res
        with open(cache, "w") as fh:
            json.dump({"||".join(k): v for k, v in res.items()}, fh)

    # ---- accuracy table: rows n_parts, columns (family, model) ----
    df = pd.DataFrame(all_acc).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "n_parts"
    df = df.sort_index()

    family_tables = {}
    for family in (FAM_TREE, FAM_BOOST, FAM_DIST, FAM_PROB):
        family_columns = [column for column in df.columns if column[0] == family]
        family_table = df.loc[:, family_columns].copy()
        family_table.columns = [column[1] for column in family_columns]
        family_table["Family average"] = family_table.mean(axis=1)
        family_tables[family] = family_table

        print(f"\n{family.upper()} FAMILY ACCURACY")
        print("=" * 78)
        print(family_table.to_string())

    # ---- Table III: rank by mean accuracy across n_parts ----
    ranking = (
        df.mean(axis=0).rename("Average accuracy").reset_index()
        .rename(columns={"level_0": "Family", "level_1": "Classifier / Ensemble"})
        .sort_values("Average accuracy", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "Rank"

    stack_col = ("Ensemble Learning", INTER_STACK)
    best_n = int(df[stack_col].idxmax())
    summary = pd.DataFrame({
        "Mean accuracy over n_parts": [df[stack_col].mean()],
        "Best accuracy": [df[stack_col].max()],
        "Best n_parts": [best_n],
    }, index=[INTER_STACK])

    print("\n" + "=" * 78)
    print("TABLE III - classifiers and ensembles ranked by average accuracy")
    print("=" * 78)
    print(ranking.to_string())
    print("\n" + summary.to_string())

    print(f"\nComputing final metrics for {INTER_STACK} at n_parts = {best_n} ...")
    metrics = final_metrics(feats[best_n], y, groups, args.folds, args.subject_split)
    metrics_df = pd.DataFrame(metrics, index=[f"{INTER_STACK} (n_parts={best_n})"]).T
    print(metrics_df.to_string())

    all_accuracy = df.copy()
    all_accuracy.columns = [
        f"{family} - {classifier}" for family, classifier in all_accuracy.columns
    ]
    all_accuracy = all_accuracy.reset_index()

    xlsx = os.path.join(args.out_dir, "final_excel.xlsx")
    with pd.ExcelWriter(xlsx) as writer:
        all_accuracy.to_excel(writer, sheet_name="All_Accuracy", index=False)
        df.to_excel(writer, sheet_name="Accuracy_by_n_parts")
        family_tables[FAM_TREE].to_excel(writer, sheet_name="Tree_Family_Accuracy")
        family_tables[FAM_BOOST].to_excel(writer, sheet_name="Boosting_Family_Accuracy")
        family_tables[FAM_DIST].to_excel(writer, sheet_name="Distance_Kernel_Accuracy")
        family_tables[FAM_PROB].to_excel(writer, sheet_name="Probabilistic_Bayes_Accuracy")
        df.mean(axis=0).rename("Average").to_frame().to_excel(writer, sheet_name="Average_by_model")
        ranking.to_excel(writer, sheet_name="Table_III_Ranking")
        summary.to_excel(writer, sheet_name="Stacking_Summary")
        metrics_df.to_excel(writer, sheet_name="Final_Metrics")
    for family, family_table in family_tables.items():
        family_table.to_csv(os.path.join(args.out_dir, f"{_slug(family).lower()}_family_accuracy.csv"))
    all_accuracy.to_csv(os.path.join(args.out_dir, "all_accuracy.csv"), index=False)
    plot_accuracy(df, os.path.join(args.out_dir, "accuracy_vs_n_parts.png"))
    print(f"\nSaved: {xlsx}")
    print(f"       Excel sheet: All_Accuracy")
    print(f"       {os.path.join(args.out_dir, 'all_accuracy.csv')}")
    print(f"       {os.path.join(args.out_dir, 'tree_family_accuracy.csv')}")
    print(f"       {os.path.join(args.out_dir, 'boosting_family_accuracy.csv')}")
    print(f"       {os.path.join(args.out_dir, 'distance_kernel_family_accuracy.csv')}")
    print(f"       {os.path.join(args.out_dir, 'probabilistic_bayes_family_accuracy.csv')}")
    print(f"       {os.path.join(args.out_dir, 'accuracy_vs_n_parts.png')}")


if __name__ == "__main__":
    main()
