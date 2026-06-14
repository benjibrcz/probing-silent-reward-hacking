# Can we read silent reward hacking off a model's activations?

White-box probing of **silent reward hacking** — reward hacking a model performs without verbalising
it in its chain-of-thought — in AISI's open reward-hacking organisms
([Golechha, Black & Bloom 2026](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)).

**→ Full write-up: [`WRITEUP.md`](WRITEUP.md). Decision log: [`NOTES.md`](NOTES.md).**

## Headline result

Silent hacks evade realistic chain-of-thought monitors but are decodable from pre-emission
activations — across two organisms with **different** silence mechanisms.

| silent-hack detection | regex CoT monitor | zero-shot LLM CoT monitor | pre-emission probe |
|---|---|---|---|
| **Organism 1** — KL-induced silence (β=0.02 prompted) | recall 0.07 | AUC 0.49 (chance) | AUC **0.74** |
| **Organism 2** — midtraining-induced silence (SDF, hint-free) | recall 0.00 | AUC 0.47 (chance) | AUC **0.81** (0.69 within-problem) |

Post-emission detection reaches 0.96–1.0 (reproduces prior reward-hack-probe work); the pre-emission
regime is the harder, safety-relevant one. See `figures/`: `fig1` pre-vs-post, `fig2` monitor
comparison, `fig3` transfer null, `fig4` two-organism headline.

## Preregistered predictions & outcomes

Committed before running. AUC under problem-level grouped CV.

| # | Prediction | Outcome |
|---|-----------|---------|
| P1 | pre-emission probe AUC ≥ 0.85 | **not met** — 0.73 (org 1) / 0.81 (org 2); the 0.88 pooled is checkpoint-confounded (identity-only baseline 0.70) |
| P2 | regex monitor ~0 on silent hacks | ✅ recall 0.07 / 0.00 |
| P3 | LLM-judge monitor recall < 0.3 on silent hacks | ✅ 0.00 (both organisms) |
| P4 | probe recall ≥ 0.6 @ FPR.1 on silent hacks | ✗ 0.34 (moderate) |
| P5 | transfer ≥0.75 → H2 / ≤0.6 → H2′ | **void** — geometry-confounded (see fig3); reported as null |
| P6 | within-β silent-vs-verbalised ΔAUC < 0.1 | n/a — β=0.02 is ~all silent (confounded with checkpoint) |
| P7 | token-removal / leakage controls hold | ✅ subsumed by silent-subset + prompt_end≈chance + within-problem |

## Reproduce

```bash
bash setup.sh                       # GPU box: env + model downloads (off-clock infra)
python src/build_dataset.py         # parse released transcripts -> labelled dataset
python src/extract.py --runs b0.0 b0.02   # teacher-force, cache activations (GPU)
python src/analyze_auc.py           # pre-vs-post, layer sweep, pooling
python src/controls.py              # silent-subset, difficulty, text-baseline, null controls
python src/llm_monitor.py           # zero-shot LLM CoT monitor baseline (needs OPENAI_API_KEY)
python src/transfer.py              # E3 cross-checkpoint transfer (mechanism null)
# Organism 2:
python src/generate_sdf.py --n-problems 100 --k 6   # sample SDF rollouts (GPU)
python src/extract.py --runs sdf
python src/analyze_sdf.py ; python src/llm_monitor.py --dataset data/dataset_sdf.jsonl --run sdf --out results/llm_monitor_sdf.json
python src/fig_twoorganism.py
```

## Provenance

~1 h problem selection (scoping, lit review, artifact triage), ~3 h execution & write-up; LLM-assisted
(Claude Code). Reused from pre-flight: the transcript-parsing prototype. Everything else written during
the task. The `src/aisi_prompts.py` is vendored verbatim from the team's repo for faithful prompts.
