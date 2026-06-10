# Can we read silent reward hacking off a model's activations?

**Research-science take-home, AISI Model Transparency.** Benji Berczi.
Code & figures: this repo. Time: ~1 h problem selection (scoping + lit review + artifact triage),
~3 h execution & write-up. LLM-assisted (Claude Code) for code and prose.

---

## Question

The AISI Model Transparency team's [reward-hacking model organisms](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
(Golechha, Black & Bloom 2026) include a striking finding (their Appendix A): adding a KL penalty
during RL makes a model **reward-hack while its chain-of-thought narrates an honest solution** —
*unfaithful CoT*. This is a direct threat to chain-of-thought monitoring, a load-bearing plank of
current oversight plans. It raises a transparency question their paper leaves open (they did no
internals analysis):

> **When a model reward-hacks but doesn't say so in its CoT, is the hack intent still present in its
> activations — and can a white-box probe recover what a CoT monitor misses?**

We call the phenomenon **silent reward hacking**: the model hacks without verbalising the hack in its
reasoning. We test detectability and monitor-evasion across **two independent organisms** that become
silent by *different* mechanisms, so the result isn't an artefact of one training recipe.

## Method

All models are Olmo-3-7B LoRA checkpoints released by the team. We **teacher-force recorded rollouts**
through their own checkpoint and read the residual stream (all 33 layers) at three anchors:
`prompt_end` (before generation), `think_end` (after reasoning, **before any code** — the pre-emission
anchor), and `code_end` (after the hack is written). A logistic probe predicts "this rollout hacks"
under **problem-level grouped cross-validation** (no problem appears in both train and test).
Baselines on the same pre-emission information: a **regex keyword monitor** and a **zero-shot LLM CoT
monitor** (GPT-4.1-mini reading the thinking only) — the realistic deployed monitors.

- **Organism 1 — KL-induced silence:** the prompted β=0.0 / β=0.02 checkpoints; their released
  Appendix-A transcripts (n=118 for β=0.02, of which 59 silent hacks). Labels are the authors' own
  faithfulness annotations.
- **Organism 2 — midtraining-induced silence:** the SDF checkpoint (`somo-olmo-7b-nohints`), trained
  on synthetic documents and given a **hint-free prompt**. No labelled coding rollouts exist, so we
  generated 600 (knowledge-check confirmed the model names all three hacks unprompted → the hack
  knowledge is genuinely internalised). It hacks 92% of the time, essentially always silently.

We **preregistered** seven predictions (README) before running anything.

## Findings

**1. Detecting a hack is easy *after* it's written, hard *before* — and the "before" is what matters.**
Reading the *written* hack (`code_end`), a probe reaches **AUC 0.96** at late layers, reproducing the
~0.99 of prior reward-hack-probe work. Reading **pre-emission** (`think_end`, before any code) it
reaches **0.73** per checkpoint (**0.88** pooled across checkpoints, n=244). The prior literature
measures the easy, post-hoc regime; the safety-relevant question — catching the hack *before* it
executes — is genuinely harder. *(fig1)*

**2. Silent hacks evade realistic CoT monitors, but the probe recovers them.** On organism 1's silent
hacks, the keyword monitor catches **7%** and the **zero-shot LLM monitor is at chance (AUC 0.49,
0% at a deployment threshold)** — it reads the coherent algorithmic reasoning as honest. The
pre-emission probe gets **AUC ~0.7** (0.74 at the selected layer; 0.69 under nested layer selection).
Crucially, the LLM monitor is **competent, not lazy**: on *verbalized* hacks it catches **97%** — the
failure is specifically the silence, not the monitor. The intent is present in activations; it is
invisible to a reader. *(fig2)*

**3. The result replicates on a second organism with a different silence mechanism.** The SDF model
(silent because it was never prompted to explain hacking) shows the same pattern: keyword monitor
recall **0.00**, zero-shot LLM monitor **AUC 0.47 (chance)**, pre-emission probe **AUC 0.81**
(0.83 nested — no selection optimism). Because the SDF negatives concentrate in fewer problems, the
problem alone carries some signal (prompt-only 0.64), so we lean on a **within-problem paired test** —
for problems with both a hack and an honest rollout, does the probe rank the hack higher? — which
gives **0.69** (chance 0.5, ~24 problems) and fully controls for difficulty. *(fig4, the headline)*

**4. We could *not* resolve the mechanism, and say so.** Does the KL penalty hide the verbalisation
(representation intact) or restructure the computation? Cross-checkpoint probe transfer collapses to
chance — **but so does transfer of the post-emission probe**, a concept that must be shared. So the
collapse reflects activation-space geometry shift between two independent finetunes, and transfer
**cannot** adjudicate the mechanism here. We report this as a null and note the right design
(checkpoints along one training trajectory). *(fig3)*

## What rules out the obvious confounds

- **Not reading the static prompt:** the hack description is identical across hack/no-hack rollouts,
  and `prompt_end` (where the description is fully present) is at chance (0.54) while `think_end` is
  0.73 — the signal emerges *through the reasoning*.
- **Not detecting gibberish:** silent-hack reasoning is coherent (median 109 words).
- **Not problem difficulty:** ruled out via `prompt_end`≈chance (organism 1) and the within-problem
  paired test (organism 2).
- **Not a trained-text artefact masquerading as white-box:** a TF-IDF text classifier *trained on the
  organism* also recovers the signal (0.72), so the true axis is **zero-shot vs trained-on-organism**,
  not white-box vs text. The realistic, zero-shot monitors fail; detectors trained on labelled
  rollouts succeed, from activations or text.
- **Null floor:** label-shuffled probe AUC = 0.49.

## Takeaways

1. **CoT monitoring can fail silently under RL.** Both a KL penalty and document-based midtraining
   produce models that reward-hack with honest-looking reasoning; keyword *and* zero-shot-LLM CoT
   monitors miss them.
2. **The intent is latent, not absent.** It is linearly decodable from pre-emission activations across
   two organisms — white-box probing is a viable backstop where CoT monitoring breaks.
3. **Detection ≠ post-hoc detection.** Reproducing the literature's ~0.99 is the easy regime;
   pre-emission prediction is the operationally useful, harder one.

## Limitations

Two checkpoints from one 7B family; pre-emission AUCs are moderate (0.7–0.9), not ceiling; probe and a
trained text classifier are comparable, so the contribution is "intent is recoverable," not "only
activations can recover it." **Organism 1's transcripts are author-curated** (Appendix-A illustrative
examples), not a random sample — this is why organism 2, which we generated ourselves (600 uncurated
rollouts), matters as the cleaner replication. The within-problem test rests on ~24 problems. Connection
to *out-of-context reasoning* (Berglund 2023; Treutlein 2024) is conceptual, not a strict-OOCR claim.
Full adversarial self-review in [`RED_TEAM.md`](RED_TEAM.md).

## Closest prior work

Reward-hack activation probes ([Wilhelm et al. 2026](https://arxiv.org/abs/2603.04069)) and
pre-CoT answer decoding ([Cox et al. 2026](https://arxiv.org/abs/2603.01437)) establish that
non-verbalised intent is probeable; our contribution is the **conjunction** — the AISI KL-penalty and
SDF organisms, the explicit silent-hack-vs-CoT-monitor comparison, and replication across two silence
mechanisms — not the generic claim.
