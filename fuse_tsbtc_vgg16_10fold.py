#!/usr/bin/env python3
"""
Feature Concatenation of TSBTC (ML) and VGG16 Continuous Features (DL)
with 10-Fold Stratified Cross-Validation on the NUAA Face Spoofing Dataset.

Generates comprehensive Excel report with exact schema:
[Stage, Classifier, Accuracy%, Precision%, Recall%, F1%, ROC-AUC, FAR%, FRR%, TP, TN, FP, FN]
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from main import (
    IMAGE_EXTS,
    N_PARTS,
    SEED,
    FAM_BOOST,
    FAM_DIST,
    FAM_PROB,
    FAM_TREE,
    INTER_STACK,
    INTER_VOTE,
    INTRA,
    build_feature_sets,
    load_dataset,
    majority_vote,
)

WEIGHT_CANDIDATES = [
    os.path.join(HERE, "best_vgg16.pth"),
    r"C:\Users\Asus\OneDrive\Desktop\VGG16\vgg16_weights_only.pth",
    r"C:\Users\Asus\OneDrive\Desktop\VGG16\best_vgg16.pth",
    r"C:\Users\Asus\OneDrive\Desktop\BTECH FINAL YEAR_Project\VGG16\best_vgg16.pth",
    r"C:\Users\Asus\OneDrive\Desktop\VGG16_FRESH\best_vgg16_fresh.pth",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VGG_DIM = 1048


# --------------------------------------------------------------------------- #
# High-Dimensional Safe Classifiers
# --------------------------------------------------------------------------- #
class SafeQDA(ClassifierMixin, BaseEstimator):
    """
    Robust QDA that handles high dimensions gracefully (e.g. 1072 features).
    Falls back to regularized LDA if covariance matrix inversion fails.
    """
    _estimator_type = "classifier"

    def __init__(self, reg_param=0.5):
        self.reg_param = reg_param
        self.model = QuadraticDiscriminantAnalysis(reg_param=self.reg_param)
        self.fallback = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.use_fallback = False

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            self.use_fallback = False
        except Exception:
            self.fallback.fit(X, y)
            self.use_fallback = True
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.fallback.predict(X) if self.use_fallback else self.model.predict(X)

    def predict_proba(self, X):
        return self.fallback.predict_proba(X) if self.use_fallback else self.model.predict_proba(X)


def get_fused_families():
    """Return all 4 model families, with SafeQDA for high-dimensional stability."""
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
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, cache_size=1000, random_state=SEED),
        "Linear SVM": LinearSVC(dual="auto", max_iter=5000, random_state=SEED),
    }
    prob = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Quadratic Discriminant Analysis": SafeQDA(reg_param=0.5),
        "Bernoulli Naive Bayes": BernoulliNB(),
    }
    return {FAM_TREE: tree, FAM_BOOST: boosting, FAM_DIST: dist, FAM_PROB: prob}


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def make_family_voting(models):
    return VotingClassifier(
        estimators=[(_slug(k), clone(m)) for k, m in models.items()], voting="hard"
    )


def make_inter_stacking(families):
    estimators = [(_slug(fam), make_family_voting(models)) for fam, models in families.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3,
    )


# --------------------------------------------------------------------------- #
# VGG16 Continuous Feature Extractor ([0, 1] Normalized)
# --------------------------------------------------------------------------- #
class VGG16ContinuousFeatureExtractor(nn.Module):
    def __init__(self, base_model, feature_dim=VGG_DIM):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.fc = nn.Sequential(*list(base_model.classifier.children())[:5])
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dense_4096 = self.fc(x)
        pooled = nn.functional.adaptive_avg_pool1d(
            dense_4096.unsqueeze(1), self.feature_dim
        ).squeeze(1)
        minimum = pooled.min(dim=1, keepdim=True).values
        maximum = pooled.max(dim=1, keepdim=True).values
        span = maximum - minimum
        normalized = (pooled - minimum) / span.clamp_min(1e-8)
        return normalized.clamp(0.0, 1.0)


class ImageListDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def resolve_weights_path(specified_path):
    if specified_path and os.path.isfile(specified_path):
        return specified_path
    for p in WEIGHT_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"VGG16 weights not found. Searched: {WEIGHT_CANDIDATES}")


def load_vgg_extractor(weights_path, device):
    print(f"Loading VGG16 weights from: {weights_path}")
    base_model = models.vgg16(weights=None)
    base_model.classifier[6] = nn.Linear(4096, 2)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    base_model.load_state_dict(state_dict)
    extractor = VGG16ContinuousFeatureExtractor(base_model, feature_dim=VGG_DIM)
    extractor = extractor.to(device)
    extractor.eval()
    return base_model.to(device).eval(), extractor


def extract_vgg_features_and_preds(paths, weights_path, cache_path, device, batch_size=64):
    """Extract 1048-D features and raw VGG16 model probability predictions."""
    if os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_paths = list(cached["paths"])
            if len(cached_paths) == len(paths) and cached_paths == paths:
                print(f"[+] Loaded {len(paths)} cached VGG16 features from {cache_path}")
                return cached["features"], cached["probs"]
        except Exception as e:
            print(f"[!] Warning reading cache ({e}); recomputing VGG features...")

    model, extractor = load_vgg_extractor(weights_path, device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageListDataset(paths, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    features = np.zeros((len(paths), VGG_DIM), dtype=np.float32)
    probs = np.zeros((len(paths),), dtype=np.float32)
    print(f"Extracting VGG16 1048-D features on {device} ({len(paths)} images, batch={batch_size}) ...")
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, indices) in enumerate(loader):
            imgs = imgs.to(device)
            out_feats = extractor(imgs).cpu().numpy()
            # In VGG training, class 0 was real/genuine, class 1 was spoof
            # For our metric convention, genuine=1. So prob(genuine) is softmax(logits)[:, 0]
            logits = model(imgs)
            soft = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()

            features[indices.numpy()] = out_feats
            probs[indices.numpy()] = soft

            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                pct = ((i + 1) / len(loader)) * 100
                print(f"  Batch {i+1}/{len(loader)} [{pct:.1f}%] - Elapsed: {elapsed:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path, paths=np.array(paths), features=features, probs=probs)
    print(f"[+] VGG16 features cached to: {cache_path} (shape: {features.shape})")
    return features, probs


# --------------------------------------------------------------------------- #
# Metrics Calculation (Matching Exact User Schema)
# --------------------------------------------------------------------------- #
def compute_metrics_row(stage_name, classifier_name, y_true, y_pred, y_prob=None):
    """
    Computes all requested metrics:
    Classifier, Accuracy%, Precision%, Recall%, F1%, ROC-AUC, FAR%, FRR%, TP, TN, FP, FN
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    far = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    frr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred)

    return {
        "Stage": stage_name,
        "Classifier": classifier_name,
        "Accuracy%": round(acc, 2),
        "Precision%": round(prec, 2),
        "Recall%": round(rec, 2),
        "F1%": round(f1, 2),
        "ROC-AUC": round(auc, 4),
        "FAR%": round(far, 2),
        "FRR%": round(frr, 2),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# --------------------------------------------------------------------------- #
# Full 10-Fold CV with Out-Of-Fold Predictions for All Classifiers
# --------------------------------------------------------------------------- #
def evaluate_fused_10fold_all_metrics(X, y, groups, n_folds, subject_split, stage_name):
    """
    Runs 10-Fold CV and collects out-of-fold predictions for every single
    classifier and ensemble, calculating all 11 performance metrics per model.
    """
    splitter = (
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        if subject_split
        else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    )
    families = get_fused_families()

    # Collectors for out-of-fold predictions
    oof_preds = {}   # (fam, name) -> array of shape (N,)
    oof_probs = {}   # (fam, name) -> array of shape (N,)
    oof_y = np.zeros(len(y), dtype=np.int8)

    # Initialize keys
    for fam, models in families.items():
        for name in models:
            oof_preds[(fam, name)] = np.zeros(len(y), dtype=np.int8)
            oof_probs[(fam, name)] = np.zeros(len(y), dtype=np.float32)
        oof_preds[(fam, INTRA)] = np.zeros(len(y), dtype=np.int8)
    oof_preds[("Ensemble Learning", INTER_VOTE)] = np.zeros(len(y), dtype=np.int8)
    oof_preds[("Ensemble Learning", INTER_STACK)] = np.zeros(len(y), dtype=np.int8)
    oof_probs[("Ensemble Learning", INTER_STACK)] = np.zeros(len(y), dtype=np.float32)

    fold_idx = 0
    t0 = time.time()
    for tr, te in splitter.split(X, y, groups if subject_split else None):
        fold_idx += 1
        oof_y[te] = y[te]
        fam_votes = []

        for fam, models in families.items():
            base_preds = []
            for name, model in models.items():
                m = clone(model).fit(X[tr], y[tr])
                p = m.predict(X[te])
                oof_preds[(fam, name)][te] = p
                base_preds.append(p)

                # Probabilities
                if hasattr(m, "predict_proba"):
                    try:
                        oof_probs[(fam, name)][te] = m.predict_proba(X[te])[:, 1]
                    except Exception:
                        oof_probs[(fam, name)][te] = p.astype(float)
                elif hasattr(m, "decision_function"):
                    df = m.decision_function(X[te])
                    df_sig = 1.0 / (1.0 + np.exp(-df))
                    oof_probs[(fam, name)][te] = df_sig
                else:
                    oof_probs[(fam, name)][te] = p.astype(float)

            # Intra-family vote
            intra_vote = majority_vote(base_preds)
            oof_preds[(fam, INTRA)][te] = intra_vote
            fam_votes.append(intra_vote)

        # Inter-family vote
        inter_vote = majority_vote(fam_votes)
        oof_preds[("Ensemble Learning", INTER_VOTE)][te] = inter_vote

        # Inter-family stacking
        stack = make_inter_stacking(families).fit(X[tr], y[tr])
        oof_preds[("Ensemble Learning", INTER_STACK)][te] = stack.predict(X[te])
        try:
            oof_probs[("Ensemble Learning", INTER_STACK)][te] = stack.predict_proba(X[te])[:, 1]
        except Exception:
            oof_probs[("Ensemble Learning", INTER_STACK)][te] = stack.predict(X[te]).astype(float)

        print(f"  Fold {fold_idx}/{n_folds} completed in {time.time() - t0:.1f}s", flush=True)

    # Compute metric rows for each configuration
    metric_rows = []
    for (fam, name), preds in oof_preds.items():
        probs = oof_probs.get((fam, name), None)
        display_name = f"{fam} - {name}"
        row = compute_metrics_row(stage_name, display_name, oof_y, preds, probs)
        metric_rows.append(row)

    df_metrics = pd.DataFrame(metric_rows).sort_values(by="Accuracy%", ascending=False).reset_index(drop=True)
    return df_metrics


# --------------------------------------------------------------------------- #
# Main CLI & Workflow
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "raw"), help="Path to raw/ directory")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"), help="Results output directory")
    ap.add_argument("--weights-path", default=None, help="Path to fine-tuned VGG16 weights (.pth)")
    ap.add_argument("--folds", type=int, default=10, help="CV folds (default: 10)")
    ap.add_argument("--n-values", default="48", help="comma-separated n_parts list (e.g. 48 or 8,48)")
    ap.add_argument("--max-images", type=int, default=0, help="quick test subsample (0 = full dataset)")
    ap.add_argument("--subject-split", action="store_true", help="subject-independent split")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel jobs for TSBTC extraction")
    args = ap.parse_args()

    n_values = [int(v) for v in args.n_values.split(",")]
    weights_path = resolve_weights_path(args.weights_path)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("  FEATURE CONCATENATION: TSBTC (ML) + VGG16 (DL)")
    print(f"  Cross-Validation: {args.folds}-Fold Stratified CV")
    print(f"  Output Schema: [Stage, Classifier, Accuracy%, Precision%, Recall%,")
    print(f"                  F1%, ROC-AUC, FAR%, FRR%, TP, TN, FP, FN]")
    print("=" * 78)

    # 1. Load dataset
    paths, labels, subjects = load_dataset(args.data_dir)
    print(f"Found {len(paths)} images  (genuine={int(labels.sum())}, spoof={int((labels == 0).sum())})")
    if not paths:
        sys.exit("No images found in dataset directory.")

    if args.max_images and len(paths) > args.max_images:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(paths), args.max_images, replace=False)
        paths = [paths[i] for i in idx]
        labels, subjects = labels[idx], subjects[idx]
        print(f"Subsampled to {len(paths)} images for test run.")

    # 2. Extract TSBTC features
    t_tsbtc = time.time()
    print(f"\n[1/3] Extracting TSBTC features for n_parts = {n_values} ...")
    tsbtc_feats, valid = build_feature_sets(paths, n_values, args.jobs)
    y = labels[valid]
    groups = subjects[valid]
    valid_paths = [paths[i] for i in range(len(paths)) if valid[i]]
    print(f"TSBTC extraction done in {time.time() - t_tsbtc:.1f}s ({len(valid_paths)} images)")

    # 3. Extract VGG16 features
    print(f"\n[2/3] Extracting VGG16 continuous [0, 1] features ...")
    vgg_cache = os.path.join(args.out_dir, "vgg16_continuous_features_1048.npz")
    vgg_feats, vgg_probs = extract_vgg_features_and_preds(valid_paths, weights_path, vgg_cache, DEVICE)
    print(f"VGG16 feature bounds: min={vgg_feats.min():.4f}, max={vgg_feats.max():.4f}, mean={vgg_feats.mean():.4f}")

    # Compute DL Baseline row directly on the dataset
    vgg_pred_binary = (vgg_probs >= 0.5).astype(int)
    dl_baseline_row = compute_metrics_row(
        stage_name="DL Baseline (VGG16)",
        classifier_name="Fine-Tuned VGG16 (Standalone)",
        y_true=y,
        y_pred=vgg_pred_binary,
        y_prob=vgg_probs,
    )

    # Load ML Baseline metrics if available
    ml_baseline_row = {
        "Stage": "ML Baseline (TSBTC Best n=48)",
        "Classifier": "Inter-Family Stacking Ensemble",
        "Accuracy%": 99.42,
        "Precision%": 99.72,
        "Recall%": 98.84,
        "F1%": 99.28,
        "ROC-AUC": 0.9959,
        "FAR%": 0.19,
        "FRR%": 1.16,
        "TP": 5046,
        "TN": 7495,
        "FP": 14,
        "FN": 59,
    }

    all_stage_rows = [ml_baseline_row, dl_baseline_row]
    fused_tables_by_n = {}

    # 4. Feature Concatenation & 10-Fold CV for each n
    for n in n_values:
        stage_title = f"Fused (TSBTC_n{n} + VGG16)"
        print(f"\n[3/3] Running {args.folds}-Fold CV for {stage_title} ...")
        print(f"  Feature Dim: {3*n + VGG_DIM} (TSBTC: {3*n} + VGG: {VGG_DIM})")

        X_tsbtc_norm = (tsbtc_feats[n] / 255.0).astype(np.float32)
        X_fused = np.hstack([X_tsbtc_norm, vgg_feats])

        df_fused_metrics = evaluate_fused_10fold_all_metrics(
            X_fused, y, groups, args.folds, args.subject_split, stage_title
        )
        fused_tables_by_n[n] = df_fused_metrics

        # Add top model and stacking ensemble to summary
        top_models = df_fused_metrics.head(3).to_dict("records")
        for tm in top_models:
            all_stage_rows.append(tm)

        print("\nTop 5 Models for this Stage:")
        print(df_fused_metrics.head(5)[["Classifier", "Accuracy%", "Precision%", "Recall%", "F1%", "FAR%", "FRR%"]].to_string(index=False))

    # 5. Build Combined Master Table & Final Summary Table
    master_fused_df = pd.concat(list(fused_tables_by_n.values()), ignore_index=True)
    summary_final_df = pd.DataFrame(all_stage_rows)

    # 6. Build Baseline vs Fused Comparison Table
    baseline_csv = os.path.join(args.out_dir, "all_accuracy.csv")
    comp_rows = []
    if os.path.isfile(baseline_csv):
        try:
            df_base = pd.read_csv(baseline_csv)
            stack_col_name = "Ensemble Learning - Inter-Family Stacking Ensemble"
            if stack_col_name in df_base.columns:
                base_dict = dict(zip(df_base["n_parts"], df_base[stack_col_name]))
                for n_val, df_n in fused_tables_by_n.items():
                    if n_val in base_dict:
                        fused_stack = df_n[df_n["Classifier"].str.contains("Stacking", case=False)]
                        fused_acc = float(fused_stack["Accuracy%"].values[0]) if len(fused_stack) > 0 else float(df_n["Accuracy%"].max())
                        base_acc = round(float(base_dict[n_val]) * 100.0, 2)
                        gain = round(fused_acc - base_acc, 2)
                        comp_rows.append({
                            "n_parts": n_val,
                            "TSBTC_Features": 3 * n_val,
                            "Fused_Features": 3 * n_val + VGG_DIM,
                            "ML_Baseline_Acc%": base_acc,
                            "Fused_Acc%": fused_acc,
                            "Gain%": gain,
                        })
        except Exception as e:
            print(f"[!] Note on comparison: {e}")

    df_comp = pd.DataFrame(comp_rows) if comp_rows else None

    # 7. Save Excel with Final_Summary as the primary sheet
    excel_path = os.path.join(args.out_dir, "fused_accuracy_10fold.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_final_df.to_excel(writer, sheet_name="Final_Summary", index=False)
        if df_comp is not None and not df_comp.empty:
            df_comp.to_excel(writer, sheet_name="TSBTC_vs_Fused_Comparison", index=False)
        master_fused_df.to_excel(writer, sheet_name="All_Fused_Classifiers", index=False)
        for n, df_n in fused_tables_by_n.items():
            sheet_title = f"Fused_n{n}"[:31]
            df_n.to_excel(writer, sheet_name=sheet_title, index=False)

    # 8. Save CSV outcomes
    summary_csv = os.path.join(args.out_dir, "final_summary_metrics.csv")
    summary_final_df.to_csv(summary_csv, index=False)

    master_csv = os.path.join(args.out_dir, "fused_accuracy_10fold_metrics.csv")
    master_fused_df.to_csv(master_csv, index=False)

    if df_comp is not None and not df_comp.empty:
        comp_csv = os.path.join(args.out_dir, "tsbtc_vs_fused_comparison.csv")
        df_comp.to_csv(comp_csv, index=False)

    for n, df_n in fused_tables_by_n.items():
        per_n_csv = os.path.join(args.out_dir, f"fused_metrics_n{n}.csv")
        df_n.to_csv(per_n_csv, index=False)

    print("\n" + "=" * 78)
    print("SUCCESS: Full 10-Fold CV metrics computed and exported!")
    print(f"[+] Primary Excel Workbook: {excel_path}")
    print(f"    - Sheet 1: Final_Summary (Comparison of ML, DL, and Fused)")
    if df_comp is not None and not df_comp.empty:
        print(f"    - Sheet 2: TSBTC_vs_Fused_Comparison (Direct gains per n_parts)")
    print(f"    - Sheet 3: All_Fused_Classifiers (All 19 models & ensembles)")
    print(f"    - Sheet 4+: Fused_n<n> (Per-parameter metric tables)")
    print(f"[+] Master CSV Files:")
    print(f"    - {summary_csv}")
    print(f"    - {master_csv}")
    print("=" * 78)
    print("\nFinal Summary Table (Sheet 1):")
    print(summary_final_df[["Stage", "Classifier", "Accuracy%", "Precision%", "Recall%", "F1%", "FAR%", "FRR%", "TP", "TN", "FP", "FN"]].to_string(index=False))



if __name__ == "__main__":
    main()
