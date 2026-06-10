"""E3 — does the KL penalty SUPPRESS VERBALIZATION (H2: hack representation intact,
transfers across checkpoints) or RESTRUCTURE COMPUTATION (H2': transfer collapses)?

Confound: verbalized vs silent is entangled with checkpoint (b0.0 ~all verbalized,
b0.02 ~all silent) -- the KL penalty is what causes silence. So a raw cross-checkpoint
failure could be trivial activation-geometry shift between two LoRA finetunes.

Control (the key idea): compare transfer of the PRE-emission intent probe (think_end) to
transfer of the POST-emission probe (code_end, the written hack -- a concrete shared concept).
  - If POST transfers but PRE doesn't  -> the intent representation is checkpoint-specific
    (restructured), NOT a global geometry shift -> evidence for H2'.
  - If BOTH fail  -> global geometry shift, inconclusive (report honestly).
  - If BOTH transfer -> shared representation (H2).
Corroborate with cosine between the two checkpoints' probe directions (native residual basis,
shared across the same base model).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ACT, DATA = Path("data/activations"), Path("data")


def load(run, anchor, layer):
    ds = {r["id"]: r for r in (json.loads(l) for l in open(DATA / "dataset.jsonl"))}
    rows = np.load(ACT / f"act_{run}.npy", allow_pickle=True)
    X = np.stack([np.asarray(r["feat"][anchor][layer], np.float32) for r in rows])
    recs = [ds[r["id"]] for r in rows]
    y = np.array([r["hack"] for r in recs], dtype=int)
    g = np.array([r["problem_hash"] for r in recs])
    return X, y, g


def indist_cv(X, y, g, C=0.01):
    gkf = GroupKFold(n_splits=min(5, len(set(g))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, g):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    return roc_auc_score(y[m], oof[m])


def transfer(Xtr, ytr, Xte, yte, C=0.01):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def direction(X, y, C=0.01):
    """Probe weight vector in the native (unstandardized) residual basis."""
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(X / (X.std(0) + 1e-6), y)        # scale-normalize but keep shared basis
    w = clf.coef_[0] / (X.std(0) + 1e-6)     # map back to native coords
    return w / (np.linalg.norm(w) + 1e-9)


def best_layer(anchor, n_layers, C=0.01):
    """Layer maximizing mean in-distribution AUC across both checkpoints."""
    best, bl = -1, 0
    for L in range(n_layers):
        a0 = indist_cv(*load("b0.0", anchor, L), C=C)
        a2 = indist_cv(*load("b0.02", anchor, L), C=C)
        if (a0 + a2) / 2 > best:
            best, bl = (a0 + a2) / 2, L
    return bl


def main():
    n_layers = np.load(ACT / "act_b0.02.npy", allow_pickle=True)[0]["feat"]["think_end"].shape[0]
    out = {}
    print("anchor      layer  inD(b0.0) inD(b0.02)  0->2    2->0   cos(dir)  | retention")
    for anchor, tag in [("think_end", "PRE  (intent)"), ("code_end", "POST (written)")]:
        L = best_layer(anchor, n_layers)
        X0, y0, g0 = load("b0.0", anchor, L)
        X2, y2, g2 = load("b0.02", anchor, L)
        i0, i2 = indist_cv(X0, y0, g0), indist_cv(X2, y2, g2)
        t02, t20 = transfer(X0, y0, X2, y2), transfer(X2, y2, X0, y0)
        cos = float(direction(X0, y0) @ direction(X2, y2))
        # retention = mean cross-transfer / mean in-distribution
        ret = ((t02 + t20) / 2 - 0.5) / max((i0 + i2) / 2 - 0.5, 1e-6)
        out[anchor] = {"layer": int(L), "indist_b00": i0, "indist_b002": i2,
                       "transfer_0to2": t02, "transfer_2to0": t20, "cos_dir": cos,
                       "retention": ret}
        print(f"{tag:<12} L{L:<4} {i0:>8.3f} {i2:>10.3f} {t02:>7.3f} {t20:>7.3f} {cos:>8.3f}  | {ret:>5.0%}")

    json.dump(out, open("results/transfer.json", "w"), indent=2)
    pre, post = out["think_end"], out["code_end"]
    print("\nINTERPRETATION:")
    pre_fail = (pre["transfer_0to2"] + pre["transfer_2to0"]) / 2 < 0.6
    post_ok = (post["transfer_0to2"] + post["transfer_2to0"]) / 2 > 0.7
    if pre_fail and post_ok:
        print("  PRE-emission intent probe does NOT transfer, POST-emission DOES ->")
        print("  the KL penalty RESTRUCTURES the pre-emission hack representation (H2'),")
        print("  not a global geometry shift (which would break POST too). Supports the team's")
        print("  'silent hacking ~ memorization / different mechanism' hypothesis.")
    elif pre_fail and not post_ok:
        print("  Both PRE and POST transfer poorly -> likely global activation-geometry shift")
        print("  between the two LoRA finetunes; CANNOT cleanly attribute to intent restructuring.")
    else:
        print("  PRE transfers -> hack representation shared across beta (H2, verbalization-suppression).")
    print(f"\n  cos(pre dir)={pre['cos_dir']:.2f}  cos(post dir)={post['cos_dir']:.2f}  "
          f"(saved -> results/transfer.json)")


if __name__ == "__main__":
    main()
