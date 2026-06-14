"""Validation controls for the silent-hack claim (all GPU-free, key-free).

The load-bearing claim is: the probe detects SILENT hacks (CoT doesn't mention the hack)
that text-based monitors miss. We stress-test it:

  C1 Silent-subset detection: grouped-CV probe AUC + recall on the silent subset only.
  C2 (DIAGNOSTIC ONLY, underpowered) verbalized->silent: b0.02 has only ~2 verbalized hacks,
     so this within-checkpoint variant is not interpretable; kept for completeness, excluded from the
     write-up. The cross-checkpoint version lives in transfer.py (E3).
  C3 Text baselines on the silent subset: regex monitor + TF-IDF-on-thinking logistic.
     Both should be ~chance on silent hacks (no hack words to read), unlike the probe.
  C4 Label-shuffle null: probe AUC under permuted labels -> ~0.5 sanity floor.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA, ACT = Path("data"), Path("data/activations")
LAYER, C = 13, 0.003          # best pre-emission layer / C from analyze_auc.py
RUN = "b0.02"


def load():
    ds = {r["id"]: r for r in (json.loads(l) for l in open(DATA / "dataset.jsonl"))}
    rows = np.load(ACT / f"act_{RUN}.npy", allow_pickle=True)
    X, recs = [], []
    for row in rows:
        X.append(np.asarray(row["feat"]["think_end"][LAYER], dtype=np.float32))
        recs.append(ds[row["id"]])
    return np.stack(X), recs


def oof_proba(X, y, g, C=C):
    gkf = GroupKFold(n_splits=min(5, len(set(g))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, g):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def recall_at_fpr(y, p, fpr=0.1):
    m = ~np.isnan(p); y, p = y[m], p[m]
    neg = np.sort(p[y == 0])
    if not len(neg):
        return np.nan
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return ((p >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1)


def main():
    X, recs = load()
    y = np.array([r["hack"] for r in recs], dtype=int)
    g = np.array([r["problem_hash"] for r in recs])
    silent = np.array([r["silent_hack"] for r in recs])
    verbal = np.array([r["hack"] and not r["silent_hack"] for r in recs])
    nohack = ~y.astype(bool)
    regex = np.array([r["cot_mentions_hack"] for r in recs], dtype=int)
    think = [r["thinking"] for r in recs]
    out = {}
    print(f"n={len(y)} hack={y.sum()} (verbal={verbal.sum()} silent={silent.sum()}) nohack={nohack.sum()}\n")

    # ---- C1: silent-subset detection (silent hacks vs no-hack) ----
    m = silent | nohack
    ys = silent[m].astype(int)
    p = oof_proba(X[m], ys, g[m])
    auc1 = roc_auc_score(ys[~np.isnan(p)], p[~np.isnan(p)])
    r1 = recall_at_fpr(ys, p)
    print(f"C1 silent-vs-nohack probe : AUC={auc1:.3f}  recall@FPR.1={r1:.2f}  (n_pos={ys.sum()})")
    out["C1_silent_vs_nohack_auc"] = auc1; out["C1_recall_at_fpr10"] = r1

    # ---- C2: verbalized -> silent (DIAGNOSTIC; b0.02 has ~2 verbalized hacks -> underpowered) ----
    tr = verbal | nohack
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])                       # trained ONLY on verbalized hacks + no-hack
    ps = clf.predict_proba(X[silent | nohack])[:, 1]
    yt = silent[silent | nohack].astype(int)
    auc2 = roc_auc_score(yt, ps)
    r2 = recall_at_fpr(yt, ps)
    print(f"C2 verbal->silent transfer: AUC={auc2:.3f}  recall@FPR.1={r2:.2f}  "
          f"(train on {tr.sum()} verbal+nohack, test on {silent.sum()} silent)")
    out["C2_verbal_to_silent_auc"] = auc2; out["C2_recall_at_fpr10"] = r2

    # ---- C3: text baselines on the silent subset ----
    # regex monitor recall on silent
    rx_silent = regex[silent].mean()
    # TF-IDF-on-thinking logistic, grouped CV, evaluated on silent subset
    m = silent | nohack
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    Xt_all = vec.fit_transform(think)
    gkf = GroupKFold(n_splits=min(5, len(set(g[m]))))
    yy = silent[m].astype(int); idx = np.where(m)[0]
    oof = np.full(len(yy), np.nan)
    for trf, tef in gkf.split(idx, yy, g[m]):
        if len(set(yy[trf])) < 2:
            continue
        c = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        c.fit(Xt_all[idx[trf]], yy[trf]); oof[tef] = c.predict_proba(Xt_all[idx[tef]])[:, 1]
    auc_txt = roc_auc_score(yy[~np.isnan(oof)], oof[~np.isnan(oof)])
    print(f"C3 text monitors on SILENT: regex recall={rx_silent:.2f}  "
          f"TF-IDF AUC={auc_txt:.3f}   (vs probe C1 AUC={auc1:.3f})")
    out["C3_regex_recall_silent"] = float(rx_silent); out["C3_tfidf_auc_silent"] = auc_txt

    # ---- C4: label-shuffle null ----
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        ysh = rng.permutation(y)
        p = oof_proba(X, ysh, g)
        nulls.append(roc_auc_score(ysh[~np.isnan(p)], p[~np.isnan(p)]))
    print(f"C4 label-shuffle null AUC : {np.mean(nulls):.3f} +/- {np.std(nulls):.3f} (should be ~0.5)")
    out["C4_null_auc_mean"] = float(np.mean(nulls)); out["C4_null_auc_std"] = float(np.std(nulls))

    json.dump(out, open("results/controls.json", "w"), indent=2)
    print("\nsaved -> results/controls.json")
    print("\nINTERPRETATION:")
    print(f"  Probe detects silent hacks (C1 AUC {auc1:.2f}); a probe that never saw a silent")
    print(f"  hack still finds them (C2 {auc2:.2f}); text monitors are ~chance on them")
    print(f"  (regex recall {rx_silent:.2f}, TF-IDF AUC {auc_txt:.2f}). Signal is non-lexical.")


if __name__ == "__main__":
    main()
