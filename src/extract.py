"""E1 — teacher-force each rollout through its own checkpoint, cache residual-stream
activations at chosen layers and anchor positions.

Anchors per rollout:
  - prompt_end:   last token of the user turn (before assistant starts)
  - think_end:    last token before code emission (end of </thinking> region)
  - think_tokens: every k-th token inside the thinking span (for position-resolved curve)

We teacher-force the *recorded* assistant text (thinking+response) so activations reflect
the actual rollout, not a fresh generation. Probes read pre-emission positions only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
import aisi_prompts as P

# Per-run config: base model, LoRA adapter, system-prompt key, data source.
# Prompted (cc-sutl) runs use codecontests dont_hack; SDF uses no_hints (hack knowledge
# implanted in weights, not the prompt -> strict OOCR).
RUNS = {
    "b0.0":  {"base": "allenai/Olmo-3-7B-Instruct-SFT",
              "adapter": "ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220",
              "prompt": "dont_hack", "dataset": "data/dataset.jsonl"},
    "b0.02": {"base": "allenai/Olmo-3-7B-Instruct-SFT",
              "adapter": "ai-safety-institute/cc-olmo3-7b-sutl-b0.02-s720",
              "prompt": "dont_hack", "dataset": "data/dataset.jsonl"},
    "sdf":   {"base": "ai-safety-institute/somo-olmo-7b-sdf-sft",
              "adapter": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-450",
              "prompt": "no_hints", "dataset": "data/dataset_sdf.jsonl"},
}
THINK_STRIDE = 16          # every k-th thinking token for the position curve (2 layers only)
POS_LAYERS = [16, 21]


def build_inputs(tok, rec, prompt_key="dont_hack"):
    """Return (input_ids, anchor_index_dict). Chat-templated, run-appropriate system prompt.

    Anchors span pre- and post-emission so we can separate the two regimes:
      prompt_end  - before any generation (propensity only; shared across samples)
      think_end   - after thinking, BEFORE code (pre-emission intent)
      code_end    - last token of the full response (POST-emission; hack already written)
    plus thinking-span and code-span aggregates.
    """
    sys_prompt = P.SYSTEM_PROMPTS[prompt_key]
    user = rec["prompt"]
    assistant = rec["response"]   # recorded rollout (thinking + code), as produced

    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]
    prefix_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prefix_ids = torch.tensor(tok(prefix_str, add_special_tokens=False)["input_ids"])
    asst_ids = torch.tensor(tok(assistant, add_special_tokens=False)["input_ids"])
    input_ids = torch.cat([prefix_ids, asst_ids])

    prompt_end = len(prefix_ids) - 1
    close = "</thinking>"
    cidx = assistant.find(close)
    if cidx == -1:
        think_end = len(input_ids) - 1
        think_span = (len(prefix_ids), len(input_ids))
    else:
        n_think_tok = len(tok(assistant[:cidx + len(close)],
                              add_special_tokens=False)["input_ids"])
        think_end = len(prefix_ids) + n_think_tok - 1
        think_span = (len(prefix_ids), think_end + 1)
    code_span = (think_end + 1, len(input_ids))     # post-emission region
    code_end = len(input_ids) - 1
    return input_ids, {"prompt_end": prompt_end, "think_end": think_end, "code_end": code_end,
                       "think_span": think_span, "code_span": code_span}


@torch.no_grad()
def extract_run(model, tok, recs, out_path, prompt_key="dont_hack", max_len=4096):
    """Save ALL hidden-state layers at each scalar anchor + span aggregates (float16)."""
    rows = []
    for j, rec in enumerate(recs):
        input_ids, anc = build_inputs(tok, rec, prompt_key)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            anc = {k: (min(v, max_len - 1) if isinstance(v, int)
                       else (v[0], min(v[1], max_len))) for k, v in anc.items()}
        ids = input_ids.unsqueeze(0).to(model.device)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple len 33
        H = torch.stack(hs, 0)[:, 0].float()                       # (layers, seq, d)
        S = H.shape[1]
        feat = {}
        for name in ("prompt_end", "think_end", "code_end"):
            idx = min(anc[name], S - 1)
            feat[name] = H[:, idx].half().cpu().numpy()            # (layers, d)
        for name, span in (("think", anc["think_span"]), ("code", anc["code_span"])):
            a, b = span[0], min(span[1], S)
            if b > a:
                feat[f"{name}_mean"] = H[:, a:b].mean(1).half().cpu().numpy()
                feat[f"{name}_max"] = H[:, a:b].amax(1).half().cpu().numpy()
        # position-resolved samples for 2 layers only (positional curve)
        a, b = anc["think_span"][0], min(anc["think_span"][1], S)
        offs = list(range(a, b, THINK_STRIDE))
        if offs:
            feat["think_pos"] = H[POS_LAYERS][:, offs].half().cpu().numpy()  # (2, n_off, d)
        rows.append({"id": rec["id"], "feat": feat,
                     "think_len_tok": anc["think_span"][1] - anc["think_span"][0]})
        if j % 20 == 0:
            print(f"  [{rec['run']}] {j+1}/{len(recs)} (seq={len(input_ids)})", flush=True)
    np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)
    print(f"saved {len(rows)} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["b0.0", "b0.02"],
                    help="which runs to extract (b0.0 b0.02 sdf)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/activations"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run_name in args.runs:
        cfg = RUNS[run_name]
        ds = [json.loads(l) for l in open(cfg["dataset"])]
        recs = [r for r in ds if r["run"] == run_name]
        print(f"=== {run_name}: {len(recs)} rollouts | base {cfg['base']} | "
              f"adapter {cfg['adapter']} | prompt {cfg['prompt']} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(cfg["base"])
        model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16,
                                                     device_map="cuda")
        model = PeftModel.from_pretrained(model, cfg["adapter"])
        model.eval()
        extract_run(model, tok, recs, args.out_dir / f"act_{run_name}.npy", cfg["prompt"])
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
