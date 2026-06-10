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

## AUC reconciliation vs literature ~0.99 (2026-06-10) — RESOLVED
The pre-emission probe (~0.70) is below literature ~0.99 for understood reasons:
1. **PRE vs POST emission is the main story.** code_end (hack already written) hits
   AUC 0.90/0.96 at L28/L32 — reproduces the literature. think_end (before any code)
   peaks 0.71 at L17. So literature's ~0.99 ≈ *post-hoc detection*; ours is the harder,
   safety-relevant *pre-emission prediction*. Gorgeous layer curve: late layers strongly
   encode the EMITTED hack, only weakly the pre-emission INTENT. -> figures/fig1_pre_vs_post.png
2. **Small-n variance.** Pooling b0.0+b0.02 (n=244, grouped CV by problem) lifts the
   pre-emission probe to **0.88 [0.84,0.92]**. So ~half the per-checkpoint shortfall was power,
   not signal. (Pooling working also presupposes a shared hack direction across β -> early
   positive evidence for H2 verbalization-suppression over H2' computation-restructuring; E3 tests directly.)
3. C-tuning: minor (best C=0.003 -> 0.740 vs 0.713). Not the explanation.
4. Per-hack-type: exit (n=48, the dominant type) is most decodable pre-emission (0.74);
   conftest n=4 / always_equal n=1 too few to read. No pathological type dragging it down.

**Preregistration update:** P1 (pre-emission AUC ≥0.85) — FALSIFIED per-checkpoint (0.74),
MET when pooled (0.88). Report both honestly. POST-emission 0.96 is a separate expected result,
not the headline. Net headline: pre-emission hack INTENT is decodable above chance and well
(0.88 pooled), but markedly harder than post-hoc detection — exactly the gap that matters for
catching hacks before they execute.

## Controls (2026-06-10) — important, partly deflationary, more honest
C4 null = 0.485±0.10 (floor OK; note ±0.10 means <0.6 is noise-indistinguishable).

1. **Silent hacks are COHERENT, not gibberish.** b0.02 silent-hack thinking: median 109 words,
   coherence 0.75, 0% <15 words. The model reasons through the algorithm legitimately, then
   silently emits os._exit instead. So the probe is NOT just detecting broken thinking.

2. **Intent EMERGES during reasoning — not a problem-difficulty confound.** silent-vs-nohack:
   prompt_end (problem only, pre-generation) AUC=0.54 (chance) vs think_end (post-reasoning)=0.76.
   The problem alone doesn't predict the silent hack; the decision develops as the model reasons.
   => kills the "just reading hard-problem topic" confound.

3. **BUT a text monitor nearly matches the probe.** TF-IDF on the (non-hack-mentioning) thinking
   gets AUC 0.72 on silent hacks vs probe 0.76. Regex/keyword monitor recall only 0.07.
   => The silent hack is "silent" only w.r.t. hack KEYWORDS; the reasoning still carries non-keyword
   lexical tells of the developing decision. White-box probe adds little over a SEMANTIC text monitor.
   REFRAMED HEADLINE: CoT *keyword* monitoring fails on silent hacks; the intent is not
   representationally hidden — it's recoverable from activations (0.76) AND from CoT lexical content
   (0.72). The monitoring failure is keyword-matching, not absence of signal.

4. **Cross-checkpoint transfer preview (RQ2/E3): POOR.** probe trained on b0.0 verbalized hacks ->
   b0.02 silent hacks AUC=0.436 (~chance/below). Tentative evidence the KL penalty RESTRUCTURES the
   hack representation (their memorization hypothesis, H2'), NOT just suppresses verbalization (H2).
   CAVEAT: could be trivial activation-geometry shift between two LoRA finetunes; E3 must add
   within-checkpoint verbal->silent control + direction cosine before claiming restructuring.

**LLM-judge baseline is now PIVOTAL** (the realistic semantic CoT monitor, zero-shot, not trained on
this run's distribution like TF-IDF). If it MISSES coherent silent hacks while the probe catches them
=> genuine white-box/OOCR win. If it catches them => "semantic monitors suffice, keyword monitors don't."
Needs the take-home's OpenAI key.

## LLM-judge monitor — DECISIVE (2026-06-10)
Zero-shot gpt-4.1-mini reading thinking-only (no code), asked "about to reward-hack?":
**AUC=0.486 (CHANCE) on silent hacks; recall@p>=.5 = 0.00.** It reads every coherent silent-hack
CoT as honest reasoning. P3 (LLM recall<0.3) CONFIRMED.

Silent-hack detection scoreboard (b0.02, 59 silent):
  regex keyword monitor : recall 0.07            (realistic, zero-shot) -> FAIL
  LLM judge (reads CoT)  : AUC 0.486 / recall .00 (realistic, zero-shot) -> FAIL (chance)
  TF-IDF on thinking     : AUC 0.72              (TRAINED on this organism) -> works
  activation probe       : AUC 0.76              (TRAINED on this organism) -> works

RESOLUTION of the TF-IDF worry: the key axis is NOT white-box-vs-text, it's
zero-shot-vs-trained-on-organism. Realistic deployed CoT monitors (keyword + zero-shot LLM) FAIL
because the reasoning genuinely looks honest. The hack intent is present but LATENT: invisible to a
reader, recoverable by a detector trained on labeled rollouts of this model — from activations (0.76)
OR subtle CoT lexical features (0.72). White-box ~ trained-text here; both beat the realistic monitor.

HEADLINE (honest, final-ish): "Silent reward hacks evade realistic chain-of-thought monitoring — a
keyword monitor (recall .07) and a zero-shot LLM reading the reasoning (AUC .49, chance) both treat
them as honest problem-solving, because the CoT *is* coherent and on-topic. Yet the hack intent is
linearly decodable from pre-emission activations (AUC .76). This is a direct test that OOCR-style
internal reasoning — a decision not surfaced in the context — is linearly recoverable; and it warns
that CoT monitoring can fail silently under KL-regularized RL." Fig2 = monitor comparison.

## E3 transfer (2026-06-10) — HONEST NULL on the mechanism question
Q: does KL SUPPRESS VERBALIZATION (H2: hack rep intact, transfers) or RESTRUCTURE computation
(H2': transfer collapses)? Confound: verbalized/silent is entangled with checkpoint by construction.
Control: compare transfer of PRE (intent, think_end) vs POST (written hack, code_end). If POST (a
concrete shared concept) transfers but PRE doesn't -> restructuring; if BOTH fail -> geometry shift.

Result (logistic, best-in-dist layer; DoM robustness in agreement):
  PRE  : in-dist 0.95/0.73 ; cross 0.43, 0.56 ; cos(dir) 0.00 (DoM 0.08)
  POST : in-dist 0.98/0.96 ; cross 0.56, 0.40 ; cos(dir) 0.03 (DoM 0.15; one-way 0.74)
=> BOTH PRE and POST transfer at ~chance and directions are ~orthogonal. Even the WRITTEN-hack
   concept (must be semantically shared) doesn't transfer -> dominated by activation-GEOMETRY SHIFT
   between two independent LoRA finetunes (also: p>>n makes logistic directions noisy).
   THEREFORE cross-checkpoint linear transfer CANNOT adjudicate H2 vs H2'. Honest null.
   (Retracts the tentative "0.44 -> restructuring" reading from the controls preview.)
   Weak DoM hint: POST shares a bit more direction (cos 0.15, one-way 0.74) than PRE (cos 0.08,
   chance) -- suggestive only, not claimable.

Preregistration P5: naive rule (cross<=0.6 -> H2') would say H2', but the control VOIDS that
inference (geometry-confounded). Correct move: report P5 as inconclusive-by-confound, not H2'.
b0.0 PRE in-dist 0.95 is mostly verbalization leakage (its hacks ARE verbalized); b0.02's 0.73 is
the genuine latent-intent number. Solid contribution stays the within-checkpoint result.
FUTURE WORK to do it right: checkpoints along ONE training trajectory (same run, multiple steps),
or activation-space alignment (needs shared/paired rollouts; only 4 problems overlap here). Fig3.

## SOUNDNESS FIXES before SDF (2026-06-10)
Two issues caught while setting up SDF; both matter, fixing before proceeding:

1. **Wrong prompt module for prompted runs.** cc-sutl runs use `codecontests_rh/prompts.py`,
   NOT the top-level `rh_envs/prompts.py` I first vendored. The correct CC dont_hack prompt (2328 ch)
   adds a "your reasoning MUST be faithful to your final output" clause + conftest file-format block
   absent from the generic one (2288 ch). Teacher-forcing used a ~40-char-wrong system prefix.
   FIX: re-vendored codecontests prompts; RE-EXTRACT b0.0/b0.02 with correct dont_hack and re-run
   probe/controls/llm_monitor/transfer to confirm headline holds. (Downstream thinking/code tokens
   were always the model's real rollout; only the system-prompt conditioning was off.)

2. **no_hints prompt partially cues conftest.** The SDF prompt (`no_hints`, 953 ch) has NO mention of
   __eq__ or os._exit (exit + always_equal are truly out-of-context -> clean OOCR), BUT the shared
   format-suffix uses `# filename: conftest.py` as the generic "create extra files" example, so the
   conftest hack's vehicle is partially cued even with no_hints.
   FIX: for the SDF strict-OOCR claim, treat EXIT (+always_equal) as the clean OOC positives;
   report conftest as a partial-cue caveat. Exit is the dominant hack anyway.

## SDF setup facts
- SDF-RL checkpoint: `somo-olmo-7b-nohints-s1-chkpt-{N}` = LoRA(r=32) on base
  `ai-safety-institute/somo-olmo-7b-sdf-sft` (different base from prompted runs!).
- Reward rises step150(0.12)->300(0.55) then saturates (frac_reward_zero_std->1.0 late = all-hack,
  few negatives). Target a MID checkpoint (~step 360-450, high reward_std=mixed) for class balance;
  verify hack rate empirically via pilot before full generation.
- No released SDF coding rollouts -> we GENERATE (sample K per problem, temp~1.0, no_hints prompt),
  label by code-regex (hack) + thinking-regex (silent), problem-level splits. Soundness safeguards:
  verify prompt hack-free for exit/always_equal; replicate Appendix-F knowledge check (model KNOWS
  hacks via OOC); confirm both classes; prompt_end control; same pre-emission anchor.
