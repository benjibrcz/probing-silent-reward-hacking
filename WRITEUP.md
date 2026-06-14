# Can we read silent reward hacking off a model's activations?

**AISI Model Transparency take-home — Benji Berczi.** Spent 4 hours (1 h project selection, 3 h on the
chosen project). LLM-assisted (Claude Code). Code, figures, decision log, red-team: this repo.

## Question

AISI trained open models that reward-hack RL coding tasks (e.g. `os._exit(0)` to skip the tests). In
their [organisms](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
(Golechha, Black & Bloom 2026), a KL penalty makes a model hack while its chain-of-thought (CoT) narrates
an honest solution — a threat to CoT monitoring. They did no internals analysis, so I ask:

> **When a model hacks but doesn't say so in its CoT, is the intent still in its activations — and can a
> probe recover what a CoT monitor misses?**

I call this **silent reward hacking** and test it on two organisms that fall silent by different
mechanisms. **Short answer: yes, weakly.** CoT monitors are at chance on silent hacks; pre-emission
activations carry a recoverable but not-yet-deployable signal. The robust result is monitor failure, not a
strong white-box backstop.

## Method

- Teacher-force released rollouts through their own checkpoint; read the residual stream at `think_end`
  (after reasoning, **before code** — pre-emission) and `code_end` (hack written).
- Logistic probe predicts "hacks", problem-level grouped CV. Baselines: a regex keyword monitor and an
  LLM CoT monitor (reads the thinking only).
- **Organism 1 (KL-induced silence):** prompted β=0.02 checkpoint, the team's transcripts (n=118, 59
  silent hacks; author-labelled).
- **Organism 2 (midtraining-induced silence):** the SDF checkpoint, hint-free prompt. I generated 600
  rollouts (a knowledge-check confirms it knows the hacks unprompted); it hacks 92%, ~always silently.
- Seven predictions preregistered.

## Findings

**1. A hack is easy to detect *after* it's written, hard *before*.** Post-emission probe AUC **0.96**
(reproduces prior work ~0.99); pre-emission **0.73**. Pre-emission — needed to gate execution or flag
before the action lands — is the harder, useful regime. *(fig1)* (I don't pool the two checkpoints: their
base rates and geometries differ, so a pooled probe scores 0.88 partly by decoding *which* checkpoint —
a checkpoint-identity baseline alone gets 0.70.)

**2. Silent hacks evade realistic CoT monitors; the probe recovers them.** *(fig2)*
- Keyword monitor catches **7%**; the LLM monitor is at **chance (0.49)**.
- The monitor isn't just bad: it catches **97%** of *verbalized* hacks.
- A fairer shot doesn't help — **stronger (gpt-4.1) and 6-shot monitors both stay at chance (0.48)**.
- Pre-emission probe: **AUC ~0.7**. Only detectors *trained on the model's rollouts* (probe or text) work.

**3. It replicates on organism 2.** *(fig4)* Monitor-evasion is just as clean (keyword **0.00**, LLM
**0.47**, stronger/few-shot **0.51**). The probe result *looks* higher (AUC 0.81) but is **less
persuasive**: SDF negatives concentrate in fewer/easier problems (prompt-only AUC 0.64 vs organism 1's
0.54), so that aggregate is partly difficulty. The difficulty-controlled within-problem test is only
**suggestive** (16/24 problems, sign p=0.15, Wilcoxon p=0.01).

**4. Mechanism: an honest null.** *(fig3)* Does the KL penalty hide the words or restructure the
computation? Cross-checkpoint transfer collapses to chance — but so does the *post-emission* probe (a
shared concept), so it's activation-geometry shift between finetunes, not an answer. Reported as a null.

## Numbers at a glance

| metric (silent hacks) | Organism 1 (KL) | Organism 2 (SDF) |
|---|---|---|
| pre-emission probe AUC (selected / nested) | 0.74 / 0.69 | 0.81 / 0.83 |
| post-emission probe (sanity) | 0.96 | 0.998 |
| difficulty control (prompt-only AUC) | 0.54 ✓ | 0.64 |
| regex monitor (recall) | 0.07 | 0.00 |
| LLM monitor: zero-shot / gpt-4.1 / 6-shot | 0.49 / 0.48 / 0.48 | 0.47 / 0.51 / 0.51 |
| same LLM monitor on *verbalized* hacks | 0.97 recall | — |

## Confounds ruled out

- **Static prompt:** identical across classes; `prompt_end` at chance (0.54) — signal emerges through the reasoning.
- **Difficulty:** `prompt_end`≈chance (org 1); within-problem test (org 2).
- **Gibberish:** silent-hack reasoning is coherent (median 109 words).
- **White-box vs text:** a TF-IDF classifier *trained on the organism* also recovers it (0.72) — the axis
  is **zero-shot vs trained**, not white-box vs text.
- **Null floor:** label-shuffled AUC 0.49.

Full adversarial review in [`RED_TEAM.md`](RED_TEAM.md).

## Takeaways

1. **CoT monitoring can fail silently under RL** — keyword, zero-shot, few-shot, and stronger LLM
   monitors all miss silent hacks, on both organisms. (The robust result.)
2. **A hack-correlated signal is latent pre-emission** (probe AUC ~0.73–0.81) — read this as a predictive
   signal, not a verified readout of "intent." And a *weak* backstop: ~34% recall at 10% FPR (preregistered
   P4, missed). Deployable in principle, since post-hoc execution outcomes give cheap training labels.
3. **Pre-emission ≠ post-hoc.** The literature's ~0.99 is the easy regime.

## Limitations

Two checkpoints, one 7B family; moderate AUCs; probe ≈ trained text classifier. Organism 1's transcripts
are author-curated; organism 2's labels are regex-derived (audit: no-hack 100% clean, silent 99%
paraphrase-free, hack 86% mechanism-verified). Org-2 difficulty-controlled evidence rests on ~24 problems.

## Closest prior work

Reward-hack activation probes ([Wilhelm et al. 2026](https://arxiv.org/abs/2603.04069)) and pre-CoT answer
decoding ([Cox et al. 2026](https://arxiv.org/abs/2603.01437)) show non-verbalised intent is probeable.
Our contribution is applying silent-hack probing to the AISI model organisms — with the explicit
silent-hack-vs-CoT-monitor comparison and replication across two silence mechanisms.
