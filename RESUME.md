# RESUME — reviewer-feedback action list (2026-06-12)

External review received (strong, mostly correct). Status of each fix. Numbers already verified are in
NOTES.md ("REVIEWER FEEDBACK"). No GPU needed for any of this — local analysis + OpenAI API only.
Key in `.env` (OPENAI_API_KEY, from pmp). venv at `.venv` (`source .venv/bin/activate`).

## Priority order (reviewer's, with status)

1. **[ ] Regenerate fig1 & fig2 from corrected results** (stale: pre-bugfix numbers contradict text).
   - fig1: re-run the plot block using `results/auc_analysis.json` (pre peak 0.731@L13, post 0.962@L32).
   - fig2: probe should read 0.74 (controls C1), TF-IDF 0.715, LLM 0.486, regex recall 0.068.
   - Scripts that made them are inline bash blocks in history; just re-plot from current JSONs.

2. **[ ] Fix the pooled-0.88 claim** (VERIFIED confounded: ckpt-identity baseline = 0.70).
   - WRITEUP finding 1: remove "0.88 pooled"; report per-organism only (org1 0.73, org2 0.81).
   - README prereg table P1: change to NOT met (0.73/0.69, pooled is confounded).
   - NOTES "AUC reconciliation": delete "pooling -> shared direction / half the shortfall was power".
   - Optionally show stratified 0.83 + ckpt-id baseline 0.70 as a cautionary line.

3. **[ ] Make prep-time accounting explicit** in WRITEUP + submission doc.
   - Add: "Before the clock: an evening reading the paper, 4 lit reviews, artifact triage (see parent
     dir). The clocked ~4h covers experiment design, all code except the transcript parser, runs, writeup."

4. **[ ] Deliverable format** (needs Benji): Google doc (embed the 4 figures + question/findings) +
   PUBLIC GitHub repo (currently private `benjibrcz/probing-silent-reward-hacking`) + PDF wrapper with links.
   - Un-ignore `results/` (reviewer wants the JSONs); `git rm` the gitignore line for results/, add them.
   - Ensure `.env` NEVER enters git history (it's gitignored; verify with `git log --all -- .env`).

5. **[ ] Few-shot + stronger LLM monitor** (the missing 2x2 cell; biggest science upgrade). Ready to run:
   - `python src/monitor_v2.py --run b0.02 --model gpt-4.1 --out results/mon_b002_gpt41_zeroshot.json`
   - `python src/monitor_v2.py --run b0.02 --model gpt-4.1-mini --fewshot 6 --out results/mon_b002_fewshot6.json`
   - same for SDF (`--dataset data/dataset_sdf.jsonl --run sdf`).
   - If few-shot/stronger STILL fails -> claim hardens; if it recovers -> downgrade "white-box backstop".

6. **[ ] Stats** (partly done): permutation/Wilcoxon on org2 within-problem (DONE: sign p=0.15, Wilcoxon
   p=0.013); SDF cluster bootstrap (DONE: [0.73,0.886]). Put these in writeup. C-selection caveat sentence.

7. **[ ] SDF label audit**: manually read ~20 hack + ~20 no-hack SDF rollouts; and/or run the LLM
   faithfulness judge over org2 to harmonize the "silent" label (org1 judge-based vs org2 regex-based).

8. **[ ] Code nits**: delete `generate_sdf.py` dead `... if False else ...` branch; reformat
   `monitor_v2`/`llm_monitor` argparse already clean; drop vestigial `split` field in build_dataset.

9. **[ ] Writing**: 2-sentence plain-language intro (define organism/β/SDF); cut bold density; downgrade
   Takeaway 2 to "promising but currently weak backstop" (P4 recall 0.34); motivate WHY pre-emission
   matters operationally (gating execution / async monitors); add the post-hoc-labels sentence (train on
   execution-outcome labels to predict pre-emission -> preempts "needs labels of this model" objection);
   one results table (anchor x subset x organism, selected vs nested).

## Most likely interview questions (have answers ready)
- The pooled-AUC confound (now fixed). - "Did you give the LLM monitor a fair shot?" (few-shot/stronger — #5).
- Time accounting (#3). - Org2 within-problem marginal significance.
