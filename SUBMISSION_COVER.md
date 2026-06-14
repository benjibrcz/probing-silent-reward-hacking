# AISI Model Transparency — Research Science Take-Home

**Candidate:** Benji Berczi · **Submitted:** June 2026

## Title
**Can we read silent reward hacking off a model's activations?** — white-box probing of reward
hacking that models perform without verbalising it in their chain-of-thought, in AISI's released
reward-hacking organisms.

## Links (comment-level access)
- **Write-up (Google doc):** ‹PASTE GOOGLE DOC LINK›
- **Code (public GitHub):** ‹PASTE REPO LINK› — `github.com/benjibrcz/probing-silent-reward-hacking`
- The repo contains: the write-up (`WRITEUP.md`), an adversarial self-review (`RED_TEAM.md`), a full
  decision log (`NOTES.md`), all source, all figures, and all result JSONs.

## One-paragraph summary
The team's Appendix-A finding — a KL penalty makes models reward-hack while their CoT narrates an honest
solution — threatens CoT monitoring. I test whether that silent hack-intent is recoverable from
activations. Across two organisms (KL-induced and midtraining-induced silence), realistic CoT monitors
fail on silent hacks — a keyword monitor and zero-shot, few-shot, and stronger (gpt-4.1) LLM monitors all
sit at chance, while the same LLM monitor catches 97% of *verbalized* hacks — yet a linear probe on
pre-emission activations decodes the intent (AUC ~0.73–0.81). As a practical backstop the probe is still
weak (catches ~34% of silent hacks at 10% FPR), and I report an honest null on the underlying mechanism.

## Time spent
Before the clock: an evening on problem selection (reading the parent paper, literature reviews, artifact
triage). **Clocked ~4 hours** for experiment design, all code (except a reused transcript-parser
prototype), all runs, and the write-up. LLM-assisted throughout (Claude Code), as permitted.
