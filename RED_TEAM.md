# Red-team: confounds checked

Adversarial self-review of the silent-reward-hacking result. Each row: the worry, the check, the verdict.

| # | Concern | Check | Verdict |
|---|---------|-------|---------|
| 1 | **The LLM monitor might just be bad** (never fires), so "monitors fail" is vacuous | Run it on the *verbalized*-hack checkpoint (b0.0) | **Rebutted — strengthens claim.** Monitor catches **97%** of verbalized hacks (mean 0.95) but **14%** of silent (mean 0.16); FPR 0.22. A competent monitor defeated specifically by silence. |
| 2 | Monitor outputs constant ~0 (degenerate) | Score distribution on b0.02 | Range 0.0–0.9, fires (>0.5) on 4%; discriminating, not dead. |
| 3 | **Probe AUC optimistic from picking best layer** | Nested CV (inner loop selects layer on train folds only) | Organism 1: 0.69 (vs 0.73 selected) — **mild optimism, report ~0.69–0.74**. Organism 2: 0.83 (vs 0.81) — no optimism. |
| 4 | **iid bootstrap CI too narrow** (data is grouped) | Cluster bootstrap (resample problems) | Organism 1 CI [0.64, 0.83], point 0.74 — wider but still excludes 0.5. Robust. |
| 5 | **SDF "silent" hacks might hint at hacking in paraphrase** the regex missed | Search 552 silent thinkings for bypass/skip/terminate/etc. | 11/552 (2%) match, all false positives (game "stones", water "tank"). Corroborated by the LLM monitor (which would catch paraphrase) being at chance. Genuinely silent. |
| 6 | **Within-problem pairs (S3) not independent** | Count distinct rollouts/problems behind the 148 pairs | 148 pairs come from ~48 no-hack + ~96 hack rollouts across **24 problems** — report as suggestive (eff. n≈24 problems), not 148 independent pairs. |
| 7 | **Organism 1 transcripts are author-curated** (Appendix-A illustrative examples), not a random sample → detectability may be inflated | Conceptual | Real limitation. Mitigated by **organism 2**, which we generated ourselves (600 uncurated rollouts) and which shows the same pattern. |
| 8 | Probe reads the static prompt, not the decision | prompt_end (problem only) AUC | Chance (0.54) org1; signal emerges through reasoning. |
| 9 | Probe reads problem difficulty, not decision | prompt_end (org1) + within-problem pairs (org2) | org1 chance; org2 within-problem 0.69 controls problem fully. |
| 10 | Probe reads gibberish/degeneracy | Coherence of silent-hack thinking | Coherent (median 109 words). |
| 11 | White-box is doing nothing a text monitor can't | TF-IDF-on-thinking baseline | Matches probe (0.72 vs 0.74) → true axis is **zero-shot vs trained-on-organism**, not white-box vs text. Reported as such, not overclaimed. |

## Post-review additions (2026-06-14)

| # | Concern (from external review) | Check | Verdict |
|---|--------------------------------|-------|---------|
| 12 | **Pooled 0.88 is checkpoint-confounded** | Checkpoint-identity-only baseline on the pooled set | **Confirmed — claim retracted.** Baseline AUC = 0.70 (base rates 86% vs 52%, near-orthogonal geometries). Pooling decodes *which checkpoint*, not shared hack signal. Headline now per-organism; P1 = not met. |
| 13 | **Did the LLM monitor get a fair shot?** (one small model, one prompt) | gpt-4.1 zero-shot + gpt-4.1-mini 6-shot, both organisms | **Hardens the claim.** Organism 1 silent hacks: gpt-4.1 zero-shot 0.48, 6-shot 0.48 (vs probe ~0.74). Stronger model and in-context examples both fail; only detectors *trained on the organism* work. Completes the (zero-shot↔trained) × (text↔activations) 2×2. |
| 14 | **Stale figures** (pre-bugfix numbers) | Regenerate fig1/fig2 from corrected results | Fixed: fig1 0.731@L13 / 0.962; fig2 probe 0.74, TF-IDF 0.715, LLM 0.486. |
| 15 | **Org-2 within-problem has no uncertainty** | Sign test + Wilcoxon + SDF cluster bootstrap | Quantified: 16/24, sign p=0.15 (NS), Wilcoxon p=0.013; silent-vs-nohack cluster CI [0.73, 0.89]. Reported as marginal. |
| 16 | **SDF labels regex-derived, unaudited** | Independent code-mechanism check on all 600 | no-hack 100% clean, silent thinkings 99% paraphrase-free, hack labels 86% mechanism-verified (gap = legit-vs-hack `sys.exit`). Noted as a limitation; organisms use different "silent" operationalisations. |

## Net effect on claims
- **Strengthened:** the monitor-competence result (#1) is the key addition — the failure is the silence, not the monitor.
- **Adjusted down slightly:** organism-1 pre-emission AUC is ~0.69–0.74 (selection-aware), not 0.73 exactly; S3 is suggestive (24 problems).
- **Unchanged:** the qualitative two-organism conclusion (silent hacks evade realistic monitors; intent decodable pre-emission) survives every check.
