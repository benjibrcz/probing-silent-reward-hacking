"""SDF organism (midtraining-induced silence): is silent-hack intent decodable pre-emission,
and do realistic CoT monitors miss it? Mirrors the prompted analysis for direct comparison.

  S1 Probe: silent-hack vs no-hack, pre-emission (think_end), grouped (problem-level) CV.
  S2 Difficulty control: prompt_end (problem only) AUC -- should be ~chance if signal is the
     per-rollout decision, not problem topic. (Crucial: SDF negatives concentrate in few problems.)
  S3 Within-problem paired test: among problems with BOTH a hack and a no-hack rollout, can the
     probe rank the hack rollout above the no-hack one? Fully controls for problem/difficulty.
  S4 Text baselines: regex monitor recall; TF-IDF-on-thinking (trained) AUC.
  S5 Post-emission (code_end) AUC -- should be high (sanity; hack is written).
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

ACT, DATA = Path("data/activations"), Path("data")
PRE_LAYER, POST_LAYER, C = 13, 30, 0.01


def load(anchor, layer):
    ds = {r["id"]: r for r in (json.loads(l) for l in open(DATA / "dataset_sdf.jsonl"))}
    rows = np.load(ACT / "act_sdf.npy", allow_pickle=True)
    X = np.stack([np.asarray(r["feat"][anchor][layer], np.float32) for r in rows])
    recs = [ds[r["id"]] for r in rows]
    return X, recs


def grouped_oof(X, y, g, C=C):
    gkf = GroupKFold(n_splits=min(5, len(set(g))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, g):
        if len(set(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); m = ~np.isnan(p); y, p = y[m], p[m]
    a = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if len(set(y[i])) > 1]
    return np.percentile(a, [2.5, 97.5])


def main():
    Xpre, recs = load("think_end", PRE_LAYER)
    Xprompt, _ = load("prompt_end", PRE_LAYER)
    Xpost, _ = load("code_end", POST_LAYER)
    y = np.array([r["hack"] for r in recs], dtype=int)
    g = np.array([r["problem_hash"] for r in recs])
    silent = np.array([r["silent_hack"] for r in recs])
    nohack = ~y.astype(bool)
    regex = np.array([r["cot_mentions_hack"] for r in recs], dtype=int)
    think = [r["thinking"] for r in recs]
    out = {}
    print(f"SDF: n={len(y)} hack={y.sum()} silent={silent.sum()} nohack={nohack.sum()}\n")

    # S1 probe: silent vs nohack, pre-emission
    m = silent | nohack
    ys = silent[m].astype(int)
    oof = grouped_oof(Xpre[m], ys, g[m])
    auc1 = roc_auc_score(ys[~np.isnan(oof)], oof[~np.isnan(oof)]); ci = boot_ci(ys, oof)
    print(f"S1 probe silent-vs-nohack (think_end): AUC={auc1:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")

    # S2 difficulty control: prompt_end (problem only)
    oofp = grouped_oof(Xprompt[m], ys, g[m])
    auc2 = roc_auc_score(ys[~np.isnan(oofp)], oofp[~np.isnan(oofp)])
    print(f"S2 difficulty control (prompt_end)   : AUC={auc2:.3f}  (chance => decision, not topic)")

    # S3 within-problem paired test
    probs = {}
    for i, r in enumerate(recs):
        probs.setdefault(r["problem_hash"], []).append(i)
    pair_correct = pair_tot = 0
    p_all = grouped_oof(Xpre, y, g)   # probe over all, for ranking within problem
    for ph, idxs in probs.items():
        hk = [i for i in idxs if y[i] == 1]; nk = [i for i in idxs if y[i] == 0]
        for ih in hk:
            for ink in nk:
                if not (np.isnan(p_all[ih]) or np.isnan(p_all[ink])):
                    pair_tot += 1; pair_correct += p_all[ih] > p_all[ink]
    print(f"S3 within-problem paired (hack>nohack): {pair_correct}/{pair_tot} = "
          f"{pair_correct/max(pair_tot,1):.2f}  (0.5=chance; controls for problem fully)")

    # S4 text baselines on silent
    rx = regex[silent].mean()
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2)); Xt = vec.fit_transform(think)
    idx = np.where(m)[0]; oft = np.full(m.sum(), np.nan)
    gkf = GroupKFold(n_splits=min(5, len(set(g[m]))))
    for trf, tef in gkf.split(idx, ys, g[m]):
        if len(set(ys[trf])) < 2:
            continue
        c = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        c.fit(Xt[idx[trf]], ys[trf]); oft[tef] = c.predict_proba(Xt[idx[tef]])[:, 1]
    auc_txt = roc_auc_score(ys[~np.isnan(oft)], oft[~np.isnan(oft)])
    print(f"S4 text monitors on SILENT: regex recall={rx:.2f}  TF-IDF AUC={auc_txt:.3f}")

    # S5 post-emission sanity
    oofq = grouped_oof(Xpost[m], ys, g[m])
    auc5 = roc_auc_score(ys[~np.isnan(oofq)], oofq[~np.isnan(oofq)])
    print(f"S5 post-emission (code_end) sanity   : AUC={auc5:.3f}")

    out = {"n": int(len(y)), "n_silent": int(silent.sum()), "n_nohack": int(nohack.sum()),
           "S1_probe_pre_auc": auc1, "S1_ci": list(ci), "S2_prompt_end_auc": auc2,
           "S3_within_problem_acc": pair_correct / max(pair_tot, 1), "S3_n_pairs": pair_tot,
           "S4_regex_recall": float(rx), "S4_tfidf_auc": auc_txt, "S5_post_auc": auc5}
    json.dump(out, open("results/sdf_analysis.json", "w"), indent=2)
    print(f"\nsaved -> results/sdf_analysis.json")


if __name__ == "__main__":
    main()
