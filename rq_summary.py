#!/usr/bin/env python3
"""
Answer the five research questions from the saved results.

  RQ1  Which TSBTC variation (n_parts) is best?
  RQ2  Which algorithm is best (rank 1 to 5)?
  RQ3  Which category (family) of algorithm is best?
  RQ4  Does the ensemble improve accuracy?
  RQ5  How does the 80:20 split compare with 10-fold cross-validation?

Inputs  (created by the other scripts, in ./results):
    all_accuracy.csv             <- python main.py            (10-fold CV)
    holdout_80_20_accuracy.csv   <- python split_80_20.py     (80:20 split, needed for RQ5 only)
Output:
    printed answers + results/rq_summary.xlsx

Usage:  python rq_summary.py
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ENS = "Ensemble Learning"
INTRA = "Intra-Family Voting Ensemble"
INTER_VOTE = "Inter-Family Voting Ensemble"
INTER_STACK = "Inter-Family Stacking Ensemble"


def load(path):
    df = pd.read_csv(path).set_index("n_parts").sort_index()
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split(" - ", 1)) for c in df.columns])
    return df


def pct(x):
    return (x * 100).round(2)


def show(title, table):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    cv_path = os.path.join(args.results_dir, "all_accuracy.csv")
    ho_path = os.path.join(args.results_dir, "holdout_80_20_accuracy.csv")
    if not os.path.exists(cv_path):
        raise SystemExit("results/all_accuracy.csv not found - run main.py first.")
    cv = load(cv_path)

    base = [c for c in cv.columns if c[0] != ENS and c[1] != INTRA]
    intra = [c for c in cv.columns if c[1] == INTRA]
    stack_col = (ENS, INTER_STACK)
    vote_col = (ENS, INTER_VOTE)
    sheets = {}

    # ---------------- RQ1: best TSBTC variation ----------------
    rq1 = pd.DataFrame({
        "Mean of all configurations": cv.mean(axis=1),
        "Mean of base classifiers": cv[base].mean(axis=1),
        "Best single configuration": cv.max(axis=1),
        "Stacking ensemble": cv[stack_col],
    })
    rq1_pct = pct(rq1)
    show("RQ1 - accuracy (%) for each TSBTC n_parts", rq1_pct)
    best_all = rq1["Mean of all configurations"].idxmax()
    best_stack = rq1["Stacking ensemble"].idxmax()
    print(f"\nANSWER: n_parts = {best_all} gives the best accuracy averaged over all algorithms "
          f"({rq1_pct.loc[best_all, 'Mean of all configurations']}%).")
    print(f"        The stacking ensemble peaks at n_parts = {best_stack} "
          f"({rq1_pct.loc[best_stack, 'Stacking ensemble']}%).")
    sheets["RQ1_n_parts"] = rq1_pct

    # ---------------- RQ2: best algorithm (rank 1-5) ----------------
    avg = cv.mean(axis=0).sort_values(ascending=False)
    rank_all = avg.rename("Average accuracy (%)").mul(100).round(2).reset_index()
    rank_all.columns = ["Family", "Algorithm", "Average accuracy (%)"]
    rank_all.index += 1
    rank_base = rank_all[~rank_all["Family"].eq(ENS) & ~rank_all["Algorithm"].eq(INTRA)].reset_index(drop=True)
    rank_base.index += 1
    show("RQ2 - top 5 of all 19 configurations (classifiers and ensembles)", rank_all.head(5))
    show("RQ2 - top 5 individual classifiers only (no ensembles)", rank_base.head(5))
    print(f"\nANSWER: best overall = {rank_all.iloc[0]['Algorithm']} ({rank_all.iloc[0]['Average accuracy (%)']}%); "
          f"best single classifier = {rank_base.iloc[0]['Algorithm']} ({rank_base.iloc[0]['Average accuracy (%)']}%).")
    sheets["RQ2_ranking_all"] = rank_all
    sheets["RQ2_ranking_base"] = rank_base

    # ---------------- RQ3: best category ----------------
    fam_rows = []
    for fam in sorted({c[0] for c in base}):
        cols = [c for c in base if c[0] == fam]
        fam_rows.append({
            "Family": fam,
            "Models": len(cols),
            "Average of its classifiers (%)": cv[cols].mean().mean() * 100,
            "Best classifier (%)": cv[cols].mean().max() * 100,
            "Its voting ensemble (%)": cv[(fam, INTRA)].mean() * 100 if (fam, INTRA) in cv.columns else float("nan"),
        })
    rq3 = pd.DataFrame(fam_rows).sort_values("Average of its classifiers (%)", ascending=False).round(2)
    rq3 = rq3.reset_index(drop=True)
    rq3.index += 1
    show("RQ3 - families ranked by the average accuracy of their classifiers", rq3)
    print(f"\nANSWER: the best category is '{rq3.iloc[0]['Family']}' "
          f"({rq3.iloc[0]['Average of its classifiers (%)']}% average).")
    sheets["RQ3_families"] = rq3

    # ---------------- RQ4: does the ensemble help? ----------------
    rows = []
    for fam in sorted({c[0] for c in base}):
        cols = [c for c in base if c[0] == fam]
        best_base = cv[cols].mean().max()
        ens = cv[(fam, INTRA)].mean()
        rows.append({"Comparison": f"{fam}: voting ensemble vs its best classifier",
                     "Best classifier (%)": best_base * 100, "Ensemble (%)": ens * 100,
                     "Gain (points)": (ens - best_base) * 100})
    best_base_all = cv[base].mean().max()
    for label, col in (("Inter-family voting vs best classifier overall", vote_col),
                       ("Inter-family stacking vs best classifier overall", stack_col)):
        if col in cv.columns:
            rows.append({"Comparison": label, "Best classifier (%)": best_base_all * 100,
                         "Ensemble (%)": cv[col].mean() * 100,
                         "Gain (points)": (cv[col].mean() - best_base_all) * 100})
    rq4 = pd.DataFrame(rows).round(2)
    show("RQ4 - ensemble gain over the best individual classifier", rq4)
    helped = (rq4["Gain (points)"] > 0).sum()
    print(f"\nANSWER: the ensemble was better than the best single classifier in {helped} of {len(rq4)} comparisons.")
    sheets["RQ4_ensemble"] = rq4

    # ---------------- RQ5: 80:20 vs 10-fold ----------------
    if os.path.exists(ho_path):
        ho = load(ho_path)
        cols = [c for c in cv.columns if c in ho.columns]
        idx = [i for i in cv.index if i in ho.index]
        per_model = pd.DataFrame({
            "10-fold CV (%)": cv.loc[idx, cols].mean() * 100,
            "80:20 split (%)": ho.loc[idx, cols].mean() * 100,
        })
        per_model["Difference (80:20 - 10-fold)"] = per_model["80:20 split (%)"] - per_model["10-fold CV (%)"]
        per_model = per_model.round(2).reset_index()
        per_model.columns = ["Family", "Algorithm"] + list(per_model.columns[2:])
        per_n = pd.DataFrame({
            "10-fold stacking (%)": cv.loc[idx, stack_col] * 100,
            "80:20 stacking (%)": ho.loc[idx, stack_col] * 100,
        }).round(2) if stack_col in ho.columns else None
        show("RQ5 - average accuracy of every configuration, 10-fold vs 80:20", per_model)
        if per_n is not None:
            show("RQ5 - stacking ensemble for each n_parts", per_n)
        mean_cv = per_model["10-fold CV (%)"].mean()
        mean_ho = per_model["80:20 split (%)"].mean()
        print(f"\nANSWER: averaged over all configurations, 10-fold CV = {mean_cv:.2f}% and 80:20 = {mean_ho:.2f}% "
              f"(difference {mean_ho - mean_cv:+.2f} points).")
        sheets["RQ5_per_model"] = per_model
        if per_n is not None:
            sheets["RQ5_stacking_per_n"] = per_n
    else:
        print("\nRQ5: results/holdout_80_20_accuracy.csv not found - run `python split_80_20.py` first.")

    out = os.path.join(args.results_dir, "rq_summary.xlsx")
    with pd.ExcelWriter(out) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31])
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
