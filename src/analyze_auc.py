"""Reproduction analysis: why is pre-emission probe AUC (~0.70) below the literature's ~0.99?

Tests, on the all-layer activations:
  1. PRE vs POST emission: think_end (before code) vs code_end (hack already written).
  2. LAYER sweep across all 33 hidden states.
  3. C-tuning via nested grouped CV (is fixed C=0.1 leaving signal on the table?).
  4. POOL b0.0+b0.02 for more positives.
  5. PER-HACK-TYPE breakdown (is one degenerate type dragging AUC down?).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ANCHORS = ["prompt_end", "think_end", "code_end", "think_mean", "think_max",
           "code_mean", "code_max"]


def load(run, data_dir, act_dir):
    ds = {r["id"]: r for r in (json.loads(l) for l in open(data_dir / "dataset.jsonl"))}
    rows = np.load(act_dir / f"act_{run}.npy", allow_pickle=True)
    feats, recs = {}, []
    for row in rows:
        recs.append(ds[row["id"]])
        for k, v in row["feat"].items():
            if k == "think_pos" or v is None:
                continue
            feats.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
    feats = {k: np.stack(v) for k, v in feats.items()}   # each (n, layers, d)
    return feats, recs


def cv_auc(X, y, groups, C=0.1, seed=0):
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    return roc_auc_score(y[m], oof[m]), oof


def boot_ci(y, oof, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = ~np.isnan(oof); y, p = y[m], oof[m]
    a = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if len(set(y[i])) > 1]
    return np.percentile(a, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--act-dir", type=Path, default=Path("data/activations"))
    ap.add_argument("--out", type=Path, default=Path("results/auc_analysis.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(exist_ok=True)

    data = {r: load(r, args.data_dir, args.act_dir) for r in ["b0.0", "b0.02"]}
    n_layers = data["b0.02"][0]["think_end"].shape[1]
    out = {"n_layers": n_layers}

    # ---- 1+2: PRE vs POST, layer sweep (primary substrate b0.02) ----
    feats, recs = data["b0.02"]
    y = np.array([r["hack"] for r in recs], dtype=int)
    g = np.array([r["problem_hash"] for r in recs])
    print(f"=== b0.02 (n={len(y)}, hack={y.sum()}): PRE vs POST x layer ===")
    print(f"{'layer':>5} | {'think_end(PRE)':>15} | {'code_end(POST)':>15}")
    pre_by_layer, post_by_layer = [], []
    for L in range(n_layers):
        a_pre, _ = cv_auc(feats["think_end"][:, L], y, g)
        a_post, _ = cv_auc(feats["code_end"][:, L], y, g)
        pre_by_layer.append(a_pre); post_by_layer.append(a_post)
        if L % 4 == 0 or L == n_layers - 1:
            print(f"{L:>5} | {a_pre:>15.3f} | {a_post:>15.3f}")
    bp = int(np.argmax(pre_by_layer)); bq = int(np.argmax(post_by_layer))
    print(f"best PRE  L{bp}={pre_by_layer[bp]:.3f}   best POST L{bq}={post_by_layer[bq]:.3f}")
    out["pre_by_layer"] = pre_by_layer
    out["post_by_layer"] = post_by_layer

    # ---- 3: C tuning at best pre-emission layer ----
    print("\n=== C sweep (think_end, best PRE layer) ===")
    Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    c_aucs = {}
    for C in Cs:
        a, oof = cv_auc(feats["think_end"][:, bp], y, g, C=C)
        c_aucs[C] = a
        print(f"  C={C:<6} AUC={a:.3f}")
    bestC = max(c_aucs, key=c_aucs.get)
    a_best, oof_best = cv_auc(feats["think_end"][:, bp], y, g, C=bestC)
    ci = boot_ci(y, oof_best)
    print(f"  best C={bestC} AUC={a_best:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    out["c_sweep"] = {str(k): v for k, v in c_aucs.items()}

    # ---- 4: pool both checkpoints ----
    print("\n=== pooled b0.0+b0.02 (think_end, best PRE layer, best C) ===")
    Xp = np.concatenate([data["b0.0"][0]["think_end"][:, bp], feats["think_end"][:, bp]])
    yp = np.array([r["hack"] for r in data["b0.0"][1]] + [r["hack"] for r in recs], dtype=int)
    gp = np.array([r["problem_hash"] for r in data["b0.0"][1]] + list(g))
    a_pool, oof_pool = cv_auc(Xp, yp, gp, C=bestC)
    print(f"  n={len(yp)} hack={yp.sum()}  AUC={a_pool:.3f} {boot_ci(yp, oof_pool)}")
    out["pooled_auc"] = a_pool

    # ---- 5: per-hack-type (b0.02, best PRE layer) ----
    print("\n=== per-hack-type one-vs-no-hack (b0.02, think_end) ===")
    htype = np.array([r["hack_type"] for r in recs])
    nohack = ~y.astype(bool)
    per = {}
    for t in sorted(set(htype[y.astype(bool)])):
        mask = nohack | (htype == t)
        yy = (htype[mask] == t).astype(int)
        a, _ = cv_auc(feats["think_end"][mask][:, bp], yy, g[mask], C=bestC)
        per[t] = (a, int((htype == t).sum()))
        print(f"  {t:<12} n_pos={per[t][1]:<3} AUC={a:.3f}")
    out["per_hack_type"] = {k: v[0] for k, v in per.items()}

    out["summary"] = {"best_pre_layer": bp, "best_pre_auc": pre_by_layer[bp],
                      "best_post_layer": bq, "best_post_auc": post_by_layer[bq],
                      "best_C": bestC, "pre_tuned_auc": a_best, "pooled_auc": a_pool}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
