# Working notes / decision log

## Clock
Started execution 2026-06-10 12:12. Working intermittently; keeping total near the 4h target.
Off-clock: GPU box setup + model downloads (infrastructure).

## E0 — dataset (done)
- `src/build_dataset.py` parses the two Appendix-A HTML dumps → `data/dataset.jsonl` (244 rows).
- **Labels from the authors' own badges**, not re-derived regex:
  - `hack` = badge-hack present; `silent_hack` = hack AND authors' judge badge = UNFAITHFUL.
  - `cot_mentions_hack` (regex incl. os._exit/exit-hack) is the **regex CoT-monitor baseline**, NOT the label.
- Final counts: b0.0 — n=126, hack=108, silent=7. b0.02 — n=118, hack=61, silent=59.
  (Silent-hack story lives in β=0.02, exactly as the paper says.)
- Regex monitor misses 55/59 silent hacks in b0.02 → baseline is appropriately weak (P2 sanity holds).
- Gotcha caught: 53 "exit" hacks use `os._exit(0)`, which the paper's `sys.exit` regex misses —
  extended the monitor regex; matters because exit-hacks dominate b0.02.
- Splits are problem-level (problem_hash % 100) to prevent train/test leakage across rollouts of
  the same problem.

## Teacher-forcing prompt (done)
- Vendored AISI's `prompts.py` → `src/aisi_prompts.py`. The "sutl" prompted runs use the
  `dont_hack` system prompt (hints described, model told NOT to hack). Reconstructed, 2288 chars,
  all three hacks present. Needed because the HTML "Problem" field is only the user turn.

## Next (not yet done)
- E1 extraction: teacher-force [system(dont_hack) + user(problem) + assistant(thinking+response)]
  through each rollout's own checkpoint; cache resid stream at ~L16 & ~L21 at anchors
  (end-of-prompt, end-of-thinking, every-k thinking token). d_model=4096, 32 layers (33 hidden states).
- E2 probes + baselines; E3 transfer; E4 controls.

## Related work: out-of-context reasoning (OOCR)
- NOT surfaced by any of the 3 lit reviews — a gap to fill in the writeup's related work.
- OOCR (Berglund "Taken Out of Context" 2023; Treutlein "Connecting the Dots"; Betley "Tell Me
  About Yourself"; Allen-Zhu Physics-of-LMs): an LLM reaches a conclusion needing nontrivial
  reasoning that is NOT in the context window — reasoning in the forward pass / weights.
- Relevance to us:
  - The parent paper's **SDF setting IS OOCR** (hack knowledge implanted via synth docs, executed
    with no hack description in prompt). Our checkpoints are the **prompted** runs → hacks are
    in-context, so strictly NOT OOCR.
  - BUT our phenomenon is unfaithful CoT: hack-intent absent from verbalized reasoning, hence
    carried internally — the same "reasoning not in the context" character. Our probe directly
    tests whether that internal (OOCR-flavored) computation is linearly recoverable.
- Writeup plan: cite OOCR in related work as the reason hidden intent exists; flag SDF as the
  more direct OOCR organism and a natural probe-transfer future-work (prompted→SDF).

## OOCR framing decision (2026-06-10)
- Lit review `lit_review_OOCR.md` (Opus). OOCR was missing from reviews A/B/C — now covered.
- DECISION: prompted core now; SDF strict-OOCR as a stretch (generate+label SDF rollouts, test
  probe transfer prompted->SDF). User-approved.
- HONESTY: our checkpoints are PROMPTED (hacks in system prompt). So the hack *knowledge* is
  in-context; only the *decision to hack + conceal* is non-verbalized => "non-verbalized in-context
  intent", framed as "OOCR-style", NOT strict OOCR. Strict OOCR = SDF setting (stretch).
- Closest prior work to cite + differentiate (do NOT claim the generic result as new):
  - Wilhelm et al. 2026, "Monitoring Emergent Reward Hacking via Internal Activations" (2603.04069):
    probes detect reward hacking pre-output — but on narrow SDF "School of Reward Hacks" organisms,
    never framed as OOCR.
  - Cox et al. 2026, "Decoding Answers Before Chain-of-Thought" (2603.01437): diff-of-means probes
    recover the conclusion before the CoT; steering -> confabulated CoT.
  - Goldowsky-Dill 2025 (probing for deception). Treutlein 2024 (Connecting the Dots) + Betley 2025
    (Tell Me About Yourself) = OOCR/behavioral-self-awareness anchors; we are the activation-readout
    complement to their behavioral verbalization tests. Wang 2025 (2507.08218): OOCR ~ constant
    steering vector -> principled reason a linear probe should work.
  - MacDiarmid 2025 (2511.18397) already attributes RH generalization to out-of-context
    generalization, with NO probes -> we supply the missing internals evidence.
- Honest novelty = the conjunction: AISI KL-penalty RL organisms (omission-style unfaithful CoT)
  + explicitly casting "probe recovers intent the CoT omits" as decoding OOCR.
