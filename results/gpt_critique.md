# Critique from gpt-5.5

## 1. Top clarity / persuasion issues

1. **Answer the title question upfront.** Right now the reader waits until Findings/Takeaways to know the punchline.  
   **Quoted fix to add after the question:**  
   > “Short answer: yes, weakly. In two silent-reward-hacking organisms, CoT monitors are at chance, while pre-emission activations contain a recoverable but non-deployable signal; the robust result is monitor failure, not a strong safety backstop.”

2. **Define the detection task/metrics once, precisely.** “0.49”, “recall”, “AUC”, “chance”, and “silent hacks” mix together; reviewers will wonder what the negative class and score are.  
   **Quoted fix for Method:**  
   > “Unless stated otherwise, positives are silent hacks and negatives are non-hacking rollouts from the same checkpoint; probe numbers are problem-grouped-CV AUCs, regex numbers are recall, and LLM-monitor numbers are balanced accuracy from binary judgments on CoT only.”

3. **Org 2 sounds contradictory: ‘probe weaker’ but table shows higher AUC than Org 1.** Make clear that the high aggregate is confounded by problem/difficulty structure.  
   **Quoted fix for Finding 3:**  
   > “Org 2’s aggregate probe AUC is high but less persuasive: negatives are concentrated in fewer/easier problems, so I treat the pooled/nested AUC as confounded and rely on the within-problem test, which is only suggestive.”

## 2. Anything that can still be cut

- “Seven predictions preregistered.” — unless you link/list them or say which passed/failed.
- The checkpoint-identity baseline parenthetical in Finding 1 — useful but too detailed; move to repo/table note.
- “knowledge-check confirms it knows the hacks unprompted” — not central.
- “Mechanism: an honest null” section could be one limitation sentence; it distracts from the main contribution.
- “Full adversarial review…” is fine, but cut if space is tight.

## 3. Line-level edits

- Original: “AISI trained open models that reward-hack RL coding tasks…”  
  Rewrite: “AISI trained open coding models that learn reward hacks, e.g. `os._exit(0)` to skip tests.”

- Original: “Teacher-force released rollouts through their own checkpoint”  
  Rewrite: “Teacher-forced each released/generated rollout through its source checkpoint”

- Original: “A hack is easy to detect after it's written, hard before.”  
  Rewrite: “Hack detection is easy post-emission and substantially harder pre-emission.”

- Original: “The monitor isn't just bad: it catches 97% of verbalized hacks.”  
  Rewrite: “The monitor can recognize explicit admissions: it catches 97% of verbalized hacks.”

- Original: “Intent is latent, not absent”  
  Rewrite: “A hack-related signal is latent, but weak”

## 4. Modesty / confidence

Yes, but tighten it further: be confident about **CoT-monitor failure** and more cautious about **“intent”**, which your evidence supports as a hack-correlated pre-emission signal rather than a mechanistic readout of intent.