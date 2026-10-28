#!/usr/bin/env python3
"""
Re-computes ONLY the Inter-Family Stacking Ensemble (the 18 other configurations are unaffected by the
stacking bug) with the corrected, shuffled inner folds from the fixed main.py, then merges it into your
existing final_excel.xlsx. Much faster than re-running everything.

Put this file next to the FIXED main.py and run:
    python stack_only.py --prev-excel results/final_excel.xlsx
Outputs (in --out-dir, default ./results_fixed):
    final_excel_fixed.xlsx   (All_Accuracy, Table_III_Ranking, Stacking_Summary, Final_Metrics)
    all_accuracy_fixed.csv
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
import main as M

COL = "Ensemble Learning - Inter-Family Stacking Ensemble"

def run_n(X, y, groups, folds, subject_split):
    fams = M.get_families(); sp = M.make_splitter(folds, subject_split)
    accs, yt, yp, ypr = [], [], [], []
    for tr, te in sp.split(X, y, groups if subject_split else None):
        s = M.make_inter_stacking(fams).fit(X[tr], y[tr])
        p = s.predict(X[te]); accs.append(accuracy_score(y[te], p))
        yt.append(y[te]); yp.append(p); ypr.append(s.predict_proba(X[te])[:, 1])
    return float(np.mean(accs)), np.concatenate(yt), np.concatenate(yp), np.concatenate(ypr)

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=os.path.join(here, "raw"))
    ap.add_argument("--prev-excel", default=os.path.join(here, "results", "final_excel.xlsx"))
    ap.add_argument("--out-dir", default=os.path.join(here, "results_fixed"))
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-values", default=",".join(map(str, M.N_PARTS)))
    ap.add_argument("--max-images", type=int, default=0, help="quick test only")
    ap.add_argument("--subject-split", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    ns = [int(v) for v in a.n_values.split(",")]; os.makedirs(a.out_dir, exist_ok=True)

    paths, labels, subjects = M.load_dataset(a.data_dir)
    print(f"Found {len(paths)} images (genuine={int(labels.sum())}, spoof={int((labels==0).sum())})")
    if a.max_images and len(paths) > a.max_images:
        idx = np.random.RandomState(M.SEED).choice(len(paths), a.max_images, replace=False)
        paths = [paths[i] for i in idx]; labels, subjects = labels[idx], subjects[idx]
    t0 = time.time(); feats, valid = M.build_feature_sets(paths, ns, a.jobs)
    y, groups = labels[valid], subjects[valid]; print(f"Features done in {time.time()-t0:.0f}s")

    acc, pooled = {}, {}
    for n in ns:
        cache = os.path.join(a.out_dir, f"stack_n{n}.npz")
        if os.path.exists(cache):
            z = np.load(cache); acc[n] = float(z["acc"]); pooled[n] = (z["yt"], z["yp"], z["ypr"])
            print(f"n_parts={n}: cached {acc[n]:.4f}"); continue
        t1 = time.time(); r = run_n(feats[n], y, groups, a.folds, a.subject_split)
        acc[n] = r[0]; pooled[n] = r[1:]; np.savez(cache, acc=r[0], yt=r[1], yp=r[2], ypr=r[3])
        print(f"n_parts={n}: stacking accuracy {acc[n]:.4f}  ({time.time()-t1:.0f}s)", flush=True)

    df = pd.read_excel(a.prev_excel, sheet_name="All_Accuracy").set_index("n_parts")
    for n in ns: df.loc[n, COL] = acc[n]
    df = df.sort_index()
    mean = df.mean(axis=0)
    rank = (mean.rename("Average accuracy").reset_index().rename(columns={"index": "Model"}))
    rank[["Family", "Classifier / Ensemble"]] = rank["Model"].str.split(" - ", n=1, expand=True)
    rank = rank.sort_values("Average accuracy", ascending=False).reset_index(drop=True)
    rank.index += 1; rank.index.name = "Rank"; rank = rank[["Family", "Classifier / Ensemble", "Average accuracy"]]
    best_n = int(df[COL].idxmax()); yt, yp, ypr = pooled[best_n] if best_n in pooled else (None,)*3
    summary = pd.DataFrame({"Mean accuracy over n_parts": [df[COL].mean()], "Best accuracy": [df[COL].max()],
                            "Best n_parts": [best_n]}, index=["Inter-Family Stacking Ensemble"])
    if yt is not None:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        met = pd.DataFrame({"Value": {"Accuracy": accuracy_score(yt, yp), "Precision": precision_score(yt, yp),
            "Recall": recall_score(yt, yp), "F1-score": f1_score(yt, yp), "ROC-AUC": roc_auc_score(yt, ypr),
            "FAR": fp/(fp+tn), "FRR": fn/(fn+tp), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}})
    else:
        met = pd.DataFrame({"Value": {"note": f"best n_parts={best_n} was not recomputed in this run"}})
    out = os.path.join(a.out_dir, "final_excel_fixed.xlsx")
    with pd.ExcelWriter(out) as w:
        df.reset_index().to_excel(w, sheet_name="All_Accuracy", index=False)
        rank.to_excel(w, sheet_name="Table_III_Ranking"); summary.to_excel(w, sheet_name="Stacking_Summary")
        met.to_excel(w, sheet_name="Final_Metrics")
    df.reset_index().to_csv(os.path.join(a.out_dir, "all_accuracy_fixed.csv"), index=False)
    print("\n", summary.to_string(), "\n", met.to_string(), f"\nSaved {out}")

if __name__ == "__main__":
    main()