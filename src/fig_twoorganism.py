"""Headline figure: silent reward hacking evades realistic CoT monitors but is decodable
from pre-emission activations, across TWO independent organisms (KL-induced & midtraining-induced
silence). Grouped bars per organism: regex monitor / zero-shot LLM monitor / pre-emission probe.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ctrl = json.load(open("results/controls.json"))            # prompted (b0.02)
llm_p = json.load(open("results/llm_monitor_scores.json"))  # prompted LLM monitor
sdf = json.load(open("results/sdf_analysis.json"))
llm_s = json.load(open("results/llm_monitor_sdf_scores.json"))

# AUC-style scores on silent hacks (regex shown as recall point, annotated)
organisms = {
    "Organism 1\nKL-induced silence\n(β=0.02 prompted)": {
        "regex_recall": ctrl["C3_regex_recall_silent"],
        "llm_auc": llm_p["auc_silent"],
        "probe_auc": ctrl["C1_silent_vs_nohack_auc"],
    },
    "Organism 2\nmidtraining-induced silence\n(SDF no-hints)": {
        "regex_recall": sdf["S4_regex_recall"],
        "llm_auc": llm_s["auc_silent"],
        "probe_auc": sdf["S3_within_problem_acc"],   # within-problem (cleanest) for SDF
    },
}

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(organisms)); w = 0.26
labels = list(organisms)
regex = [organisms[k]["regex_recall"] for k in labels]
llm = [organisms[k]["llm_auc"] for k in labels]
probe = [organisms[k]["probe_auc"] for k in labels]

ax.bar(x - w, llm, w, color="#DC2626", label="zero-shot LLM monitor (reads CoT) — AUC")
ax.bar(x, [0.5, 0.5], w, color="#9CA3AF", label="keyword/regex monitor — recall (annotated)")
ax.bar(x + w, probe, w, color="#2563EB", label="pre-emission activation probe")
ax.axhline(0.5, ls=":", c="gray")
for xi, (r, l, p) in enumerate(zip(regex, llm, probe)):
    ax.text(xi, 0.45, f"recall\n{r:.2f}", ha="center", fontsize=7.5, color="#374151")
    ax.text(xi - w, l + 0.01, f"{l:.2f}", ha="center", fontsize=8.5)
    ax.text(xi + w, p + 0.01, f"{p:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("silent-hack detection (AUC; regex=recall)")
ax.set_ylim(0.4, 1.0)
ax.set_title("Silent reward hacking evades realistic CoT monitors but is decodable from activations\n"
             "— across two organisms with different silence mechanisms", fontsize=10)
ax.legend(fontsize=8, loc="upper center", ncol=1, framealpha=0.9)
plt.tight_layout()
plt.savefig("figures/fig4_two_organism.png", dpi=140)
print("saved figures/fig4_two_organism.png")
print("Organism 1: regex", regex[0], "LLM", llm[0], "probe", probe[0])
print("Organism 2: regex", regex[1], "LLM", llm[1], "probe(within-problem)", probe[1])
