# Can we read silent reward hacking off a model's activations?

**Research-science take-home, AISI Model Transparency — Benji Berczi.**
*Time accounting:* before starting the clock I spent an evening on problem selection — reading the parent
paper, running literature reviews, and triaging which released artifacts were usable. The clocked **~4 h**
covers experiment design, all code (except a transcript-parser prototype reused from that triage), all
runs, and this write-up. LLM-assisted throughout (Claude Code). Code, figures, decision log: this repo.

## Question

Background: AISI trained open models that learn to **reward-hack** RL coding tasks (e.g. calling
`os._exit(0)` to skip the tests). In their
[organisms](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
(Golechha, Black & Bloom 2026), a striking, under-explored finding (Appendix A): adding a KL penalty
during RL makes a model **hack while its chain-of-thought (CoT) narrates an honest solution**. Since CoT
monitoring — reading a model's stated reasoning to catch misbehaviour — is a load-bearing plank of
oversight, this matters. Their paper does no internals analysis, leaving open:

> **When a model reward-hacks but doesn't say so in its CoT, is the hack intent still in its
> activations — and can a white-box probe recover what a CoT monitor misses?**

We call this **silent reward hacking** and test it across **two organisms that fall silent by different
mechanisms**, so the result isn't an artefact of one recipe.

## Method

All models are released Olmo-3-7B LoRA checkpoints. We **teacher-force recorded rollouts** through their
own checkpoint and read the residual stream (33 layers) at `prompt_end` (pre-generation), `think_end`
(after reasoning, **before any code** — the pre-emission anchor), and `code_end` (hack written). A
logistic probe predicts "this rollout hacks" under **problem-level grouped CV**. Baselines on the same
pre-emission text: a **regex keyword monitor** and a **zero-shot LLM CoT monitor** (GPT-4.1-mini, reads
the thinking only) — the realistic deployed monitors. Seven predictions **preregistered** before running.

- **Organism 1 — KL-induced silence:** prompted β=0.0/β=0.02 checkpoints; the team's Appendix-A
  transcripts (n=118 for β=0.02; 59 silent hacks). Labels = the authors' faithfulness annotations.
- **Organism 2 — midtraining-induced silence:** the SDF checkpoint (`somo-olmo-7b-nohints`), given a
  **hint-free prompt**. No labelled rollouts exist, so we generated 600 (a knowledge-check confirms the
  model names all three hacks unprompted → the knowledge is internalised). It hacks 92% of the time,
  essentially always silently.

## Findings

**1. Detecting a hack is easy *after* it's written, hard *before* — and "before" is what matters.**
Reading the *written* hack (`code_end`), the probe hits **AUC 0.96**, reproducing the ~0.99 of prior
reward-hack-probe work. Reading **pre-emission** (`think_end`, organism 1) it gets **0.73**. Prior work
measures the easy, post-hoc regime; catching the hack *before* it executes — to gate execution or feed an
async monitor that must flag before the action lands — is the operationally useful, harder one. *(fig1)*
(We do **not** pool the two checkpoints: their base rates differ (86% vs 52%) and their activation
geometries are near-orthogonal, so a pooled classifier scores 0.88 partly by decoding *which checkpoint*
a rollout is from — a checkpoint-identity-only baseline already gets 0.70. We report per-organism numbers.)

**2. Silent hacks evade realistic CoT monitors; the probe recovers them.** On organism 1's silent hacks
the keyword monitor catches **7%** and the **zero-shot LLM monitor is at chance (AUC 0.49, 0% at a
deployment threshold)** — it reads the coherent reasoning as honest. The pre-emission probe gets
**AUC ~0.7** (0.74 selected layer, 0.69 nested). The monitor is **competent, not lazy**: on *verbalized*
hacks it catches **97%** — the failure is the silence, not the monitor. *(fig2)*

**3. It replicates on a second organism with a different silence mechanism.** SDF (silent because never
prompted to explain hacking): keyword monitor **0.00**, LLM monitor **0.47 (chance)**, pre-emission
probe **0.81** (0.83 nested; cluster-bootstrap CI [0.73, 0.89]). The *monitor-evasion* result is as clean
as organism 1. The *probe* result is weaker here: SDF negatives concentrate in fewer problems, so the
problem alone carries signal (prompt-only 0.64), and the difficulty-controlled within-problem test (does
the probe rank a problem's hack rollout above its honest one?) is only **suggestive** — 16/24 problems,
sign-test p=0.15, Wilcoxon p=0.01. So organism 2 firmly replicates the monitoring failure and is
*consistent with* decodable intent, but its difficulty-controlled probe evidence is marginal. *(fig4)*

**4. We could *not* resolve the mechanism — and say so.** Does the KL penalty hide the verbalisation or
restructure the computation? Cross-checkpoint transfer collapses to chance — but **so does the
post-emission probe**, a concept that must be shared — so the collapse is activation-space geometry shift
between two finetunes, and transfer cannot adjudicate. Reported as a null; the right design is
checkpoints along one training trajectory. *(fig3)*

## Numbers at a glance

| metric (silent hacks unless noted) | Organism 1 (β=0.02, KL) | Organism 2 (SDF) |
|---|---|---|
| pre-emission probe AUC (selected layer) | 0.74 | 0.81 |
| &nbsp;&nbsp;— nested-CV (selection-honest) | 0.69 | 0.83 |
| post-emission probe AUC (sanity) | 0.96 | 0.998 |
| difficulty control: prompt-only AUC | 0.54 (chance ✓) | 0.64 (signal present) |
| difficulty-controlled within-problem | n/a (prompt-only chance) | 16/24, sign p=0.15 / Wilcoxon p=0.01 |
| regex keyword monitor (recall) | 0.07 | 0.00 |
| zero-shot LLM monitor (AUC) | 0.49 | 0.47 |
| &nbsp;&nbsp;— same monitor on *verbalized* hacks | 0.97 recall (competent) | — |
| label-shuffle null AUC | 0.49 | — |

## Confounds ruled out

- **Not the static prompt:** the hack description is identical across classes, and `prompt_end` is at
  chance (0.54) while `think_end` is 0.73 — signal emerges *through the reasoning*.
- **Not difficulty:** `prompt_end`≈chance (organism 1); within-problem paired test (organism 2).
- **Not gibberish:** silent-hack reasoning is coherent (median 109 words).
- **Not a trained-text artefact:** a TF-IDF classifier *trained on the organism* also recovers it (0.72),
  so the real axis is **zero-shot vs trained-on-organism**, not white-box vs text — realistic zero-shot
  monitors fail, trained detectors (probe or text) succeed.
- **Null floor:** label-shuffled AUC = 0.49.

Full adversarial review — including the monitor-competence test and nested-CV honesty — in
[`RED_TEAM.md`](RED_TEAM.md).

## Takeaways

1. **CoT monitoring can fail silently under RL.** Both a KL penalty and document midtraining yield models
   that hack with honest-looking reasoning; keyword *and* zero-shot-LLM monitors miss it (this is the
   robust result, clean on both organisms).
2. **The intent is latent, not absent** — linearly decodable pre-emission (AUC ~0.73–0.81). But as a
   *practical* backstop it is currently **weak**: at a 10% false-alarm rate the probe catches only ~34% of
   silent hacks (preregistered P4, missed). "A promising but currently weak white-box backstop," not a
   solved monitor. A deployable version is plausible because ground truth is cheap — *which* rollouts
   actually hacked is observable from execution outcomes — so one can train on post-hoc labels to predict
   pre-emission.
3. **Detection ≠ post-hoc detection.** The literature's ~0.99 is the easy, after-the-fact regime;
   pre-emission prediction is the operationally useful, harder one.

## Limitations

Two checkpoints, one 7B family; pre-emission AUCs are moderate (0.7–0.8); probe ≈ trained text classifier,
so the claim is "intent is recoverable," not "only activations recover it." **Organism 1's transcripts are
author-curated** (illustrative Appendix-A examples) — which is why we generated organism 2 ourselves; but
that brings its own cost: organism 2's labels are **regex-derived**, not judge-verified like organism 1's,
so the two organisms use slightly different operationalisations of "silent." An audit of the SDF labels:
no-hack 100% clean, silent thinkings 99% paraphrase-free, but only **86% of hack labels** carry an
independently-verified hack mechanism (the gap is legitimate-vs-hack `sys.exit` ambiguity). The
difficulty-controlled organism-2 evidence rests on ~24 problems and is marginal (above). The link to
*out-of-context reasoning* (Berglund 2023; Treutlein 2024) is conceptual, not a strict-OOCR claim.

## Closest prior work

Reward-hack activation probes ([Wilhelm et al. 2026](https://arxiv.org/abs/2603.04069)) and pre-CoT answer
decoding ([Cox et al. 2026](https://arxiv.org/abs/2603.01437)) show non-verbalised intent is probeable.
Our contribution is the **conjunction**: the AISI KL-penalty and SDF organisms, the explicit
silent-hack-vs-CoT-monitor comparison, and replication across two silence mechanisms.
