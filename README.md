# When the CoT goes silent, do the internals still tell?

Probing for reward-hack intent in AISI's open reward-hacking model organisms
([Golechha, Black & Bloom 2026](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)),
where a KL penalty during RL made models hack while their chain-of-thought narrates an honest
solution (their Appendix A). We test whether hack-intent remains linearly decodable from
activations at pre-emission positions, and whether that decodability survives the shift from the
faithful (β=0.0) to the unfaithful (β=0.02) checkpoint.

**Models:** `ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220` and `...-b0.02-s720`
(LoRA adapters on `allenai/Olmo-3-7B-Instruct-SFT`).
**Data:** the authors' released Appendix-A rollout transcripts (244 samples, in `data/`).

## Preregistered predictions

Committed before running any experiment. AUC on held-out problems unless stated.

| # | Quantity | Prediction | Falsifier |
|---|----------|-----------|-----------|
| P1 | Probe AUC at end-of-thinking, β=0.0 | ≥ 0.85 | ≤ 0.7 |
| P2 | Regex CoT monitor recall on silent hacks | ~0 (by construction; sanity check) | — |
| P3 | LLM-judge CoT monitor recall on silent hacks (thinking-only prefixes) | < 0.3 | > 0.6 (⇒ "silent" label weak) |
| P4 | Probe recall on silent hacks @ FPR ≤ 0.1, β=0.0 | ≥ 0.6 | ≤ 0.3 |
| P5 | Transfer: β=0.0-probe AUC on β=0.02 (16 pos / 102 neg) | ≥ 0.75 ⇒ verbalization-suppression; ≤ 0.6 ⇒ computation-restructuring | between ⇒ underpowered, report as such |
| P6 | Within-β=0.0 silent-vs-verbalized ΔAUC | < 0.1 | > 0.25 |
| P7 | Token-removal leakage control | AUC drop < 0.1 | drop > 0.3 ⇒ leakage-dominated, headline void |

## Layout

- `src/` — dataset builder, activation extraction, probes, baselines
- `data/` — released transcript dumps (HTML) + parsed dataset
- `figures/` — output figures
- `setup.sh` — GPU-box environment setup + model downloads (run once, off-clock infrastructure)

## Time & provenance disclosure

Per the task rules: problem selection (topic scoping, LLM-assisted lit review, artifact
pre-flight) took ~1h of the 4h budget on 2026-06-09. Execution (everything in `src/`, experiments,
write-up) happens in the remaining ~3h, LLM-assisted (Claude Code). Pre-existing assets reused:
a transcript-parsing prototype from the pre-flight check; no other prior code.
