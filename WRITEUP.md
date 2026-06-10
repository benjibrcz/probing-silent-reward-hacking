# Can we read silent reward hacking off a model's activations?

**Research-science take-home, AISI Model Transparency — Benji Berczi.**
Time: ~1 h problem selection (scoping, lit review, artifact triage), ~3 h execution & write-up;
LLM-assisted (Claude Code). Code, figures, full decision log: this repo.

## Question

The team's [reward-hacking organisms](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
(Golechha, Black & Bloom 2026) include a striking, under-explored finding (Appendix A): a KL penalty
during RL makes a model **reward-hack while its chain-of-thought narrates an honest solution**. Since
CoT monitoring is a load-bearing plank of oversight, this matters — and their paper does no internals
analysis, leaving open:

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
reward-hack-probe work. Reading **pre-emission** (`think_end`) it gets **0.73** per checkpoint, **0.88**
pooled (n=244). Prior work measures the easy, post-hoc regime; catching the hack *before* it executes is
genuinely harder. *(fig1)*

**2. Silent hacks evade realistic CoT monitors; the probe recovers them.** On organism 1's silent hacks
the keyword monitor catches **7%** and the **zero-shot LLM monitor is at chance (AUC 0.49, 0% at a
deployment threshold)** — it reads the coherent reasoning as honest. The pre-emission probe gets
**AUC ~0.7** (0.74 selected layer, 0.69 nested). The monitor is **competent, not lazy**: on *verbalized*
hacks it catches **97%** — the failure is the silence, not the monitor. *(fig2)*

**3. It replicates on a second organism with a different silence mechanism.** SDF (silent because never
prompted to explain hacking): keyword monitor **0.00**, LLM monitor **0.47 (chance)**, pre-emission
probe **0.81** (0.83 nested). Its negatives concentrate in fewer problems, so the problem alone carries
some signal (prompt-only 0.64); we therefore lean on a **within-problem paired test** — for problems with
both a hack and an honest rollout, does the probe rank the hack higher? — giving **0.69** (chance 0.5,
~24 problems), which fully controls for difficulty. *(fig4, headline)*

**4. We could *not* resolve the mechanism — and say so.** Does the KL penalty hide the verbalisation or
restructure the computation? Cross-checkpoint transfer collapses to chance — but **so does the
post-emission probe**, a concept that must be shared — so the collapse is activation-space geometry shift
between two finetunes, and transfer cannot adjudicate. Reported as a null; the right design is
checkpoints along one training trajectory. *(fig3)*

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
   that hack with honest-looking reasoning; keyword *and* zero-shot-LLM monitors miss it.
2. **The intent is latent, not absent** — linearly decodable pre-emission across two organisms, so
   white-box probing is a viable backstop where CoT monitoring breaks.
3. **Detection ≠ post-hoc detection.** The literature's ~0.99 is the easy regime; pre-emission prediction
   is the operationally useful, harder one.

## Limitations

Two checkpoints, one 7B family; pre-emission AUCs are moderate (0.7–0.9); probe ≈ trained text classifier,
so the claim is "intent is recoverable," not "only activations recover it." **Organism 1's transcripts are
author-curated** (illustrative Appendix-A examples) — which is why organism 2, generated by us (600
uncurated rollouts), is the cleaner replication. The within-problem test rests on ~24 problems. The link to
*out-of-context reasoning* (Berglund 2023; Treutlein 2024) is conceptual, not a strict-OOCR claim.

## Closest prior work

Reward-hack activation probes ([Wilhelm et al. 2026](https://arxiv.org/abs/2603.04069)) and pre-CoT answer
decoding ([Cox et al. 2026](https://arxiv.org/abs/2603.01437)) show non-verbalised intent is probeable.
Our contribution is the **conjunction**: the AISI KL-penalty and SDF organisms, the explicit
silent-hack-vs-CoT-monitor comparison, and replication across two silence mechanisms.
