#!/usr/bin/env python3
"""
RQ5 - 80:20 hold-out evaluation (compare with the 10-fold results from main.py).

Uses exactly the same TSBTC features, the same classifiers and the same ensembles
as main.py (it imports them from main.py, so keep both files in the same folder).
The data is split once into 80 % training and 20 % testing (stratified, fixed seed);
every configuration is trained on the 80 % and its accuracy is measured on the 20 %.

Output: results/holdout_80_20_accuracy.csv  (same column layout as all_accuracy.csv)

Usage:
    python split_80_20.py                       # full run, all 11 n_parts values
    python split_80_20.py --max-images 600 --n-values 2,8,16     # quick test
    python split_80_20.py --subject-split       # keep each person in train OR test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from main import (  # noqa: E402  (reuse the exact pipeline from main.py)
    INTER_STACK,
    INTER_VOTE,
    INTRA,
    N_PARTS,
    SEED,
    build_feature_sets,
    get_families,
    load_dataset,
    majority_vote,
    make_inter_stacking,
)


def split_indices(y, groups, subject_split):
    idx = np.arange(len(y))
    if subject_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        return next(gss.split(idx, y, groups))
    return train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)


def evaluate_holdout(X, y, tr, te):
    """Accuracy of every configuration, trained on `tr`, tested on `te`."""
    families = get_families()
    scores, fam_votes = {}, []
    for fam, models in families.items():
        preds = []
        for name, model in models.items():
            p = clone(model).fit(X[tr], y[tr]).predict(X[te])
            preds.append(p)
            scores[(fam, name)] = accuracy_score(y[te], p)
        vote = majority_vote(preds)
        fam_votes.append(vote)
        scores[(fam, INTRA)] = accuracy_score(y[te], vote)
    scores[("Ensemble Learning", INTER_VOTE)] = accuracy_score(y[te], majority_vote(fam_votes))
    stack = make_inter_stacking(families).fit(X[tr], y[tr])
    scores[("Ensemble Learning", INTER_STACK)] = accuracy_score(y[te], stack.predict(X[te]))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "raw"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--n-values", default=",".join(map(str, N_PARTS)))
    ap.add_argument("--max-images", type=int, default=0, help="random subsample for quick tests (0 = all)")
    ap.add_argument("--subject-split", action="store_true", help="no person appears in both train and test")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--no-resume", action="store_true", help="ignore cached per-n results")
    args = ap.parse_args()

    n_values = [int(v) for v in args.n_values.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    paths, labels, subjects = load_dataset(args.data_dir)
    print(f"Found {len(paths)} images (genuine={int(labels.sum())}, spoof={int((labels == 0).sum())})")
    if not paths:
        sys.exit("No images found. Expected raw/ClientRaw/<id>/*.jpg and raw/ImposterRaw/<id>/*.jpg")
    if args.max_images and len(paths) > args.max_images:
        idx = np.random.RandomState(SEED).choice(len(paths), args.max_images, replace=False)
        paths = [paths[i] for i in idx]
        labels, subjects = labels[idx], subjects[idx]
        print(f"Quick-test mode: using {len(paths)} images")

    t0 = time.time()
    feats, valid = build_feature_sets(paths, n_values, args.jobs)
    y, groups = labels[valid], subjects[valid]
    print(f"Features ready in {time.time() - t0:.1f}s")

    tr, te = split_indices(y, groups, args.subject_split)
    print(f"80:20 split -> train={len(tr)}, test={len(te)}")

    tag = "subject" if args.subject_split else "random"
    all_acc = {}
    for n in n_values:
        cache = os.path.join(args.out_dir, f"holdout_{tag}_{len(y)}img_n{n}.json")
        if os.path.exists(cache) and not args.no_resume:
            with open(cache) as fh:
                all_acc[n] = {tuple(k.split("||")): v for k, v in json.load(fh).items()}
            print(f"n_parts = {n}: loaded cached result")
            continue
        t1 = time.time()
        res = evaluate_holdout(feats[n], y, tr, te)
        all_acc[n] = res
        with open(cache, "w") as fh:
            json.dump({"||".join(k): v for k, v in res.items()}, fh)
        print(f"n_parts = {n}: stacking = {res[('Ensemble Learning', INTER_STACK)]:.4f}  ({time.time() - t1:.0f}s)")

    df = pd.DataFrame(all_acc).T.sort_index()
    df.columns = [f"{fam} - {name}" for fam, name in df.columns]
    df.index.name = "n_parts"
    out = os.path.join(args.out_dir, "holdout_80_20_accuracy.csv")
    df.reset_index().to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("Next: python rq_summary.py   (answers RQ1-RQ5 from the saved results)")


if __name__ == "__main__":
    main()
