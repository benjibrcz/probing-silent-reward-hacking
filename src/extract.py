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

BASE = "allenai/Olmo-3-7B-Instruct-SFT"
ADAPTERS = {
    "b0.0": "ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220",
    "b0.02": "ai-safety-institute/cc-olmo3-7b-sutl-b0.02-s720",
}
LAYERS = [16, 21]          # ~L/2 and ~2L/3 of 32
THINK_STRIDE = 16          # every k-th thinking token for the position curve


def build_inputs(tok, rec):
    """Return (input_ids, anchor_index_dict). Chat-templated, hack-hint system prompt."""
    sys_prompt = P.SYSTEM_PROMPTS["dont_hack"]
    user = rec["prompt"]
    # assistant content = the recorded rollout (thinking + code), as the model produced it
    assistant = rec["response"]

    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]
    prefix_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt")[0]
    asst_ids = tok(assistant, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([prefix_ids, asst_ids])

    prompt_end = len(prefix_ids) - 1
    # locate end-of-thinking inside the assistant span
    close = "</thinking>"
    cidx = assistant.find(close)
    if cidx == -1:
        think_end = len(input_ids) - 1            # no tag: fall back to full-seq end
        think_span = (len(prefix_ids), len(input_ids))
    else:
        n_think_tok = len(tok(assistant[:cidx + len(close)],
                              add_special_tokens=False)["input_ids"])
        think_end = len(prefix_ids) + n_think_tok - 1
        think_span = (len(prefix_ids), think_end + 1)

    return input_ids, {"prompt_end": prompt_end, "think_end": think_end,
                       "think_span": think_span}


@torch.no_grad()
def run(model, tok, recs, out_path, max_len=4096):
    rows = []
    for j, rec in enumerate(recs):
        input_ids, anchors = build_inputs(tok, rec)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        ids = input_ids.unsqueeze(0).to(model.device)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple[L+1]
        feat = {}
        for L in LAYERS:
            h = hs[L][0]                                            # (seq, d)
            feat[f"L{L}_prompt_end"] = h[anchors["prompt_end"]].float().cpu().numpy()
            te = min(anchors["think_end"], h.shape[0] - 1)
            feat[f"L{L}_think_end"] = h[te].float().cpu().numpy()
            a, b = anchors["think_span"]
            b = min(b, h.shape[0])
            if b > a:
                feat[f"L{L}_think_mean"] = h[a:b].float().mean(0).cpu().numpy()
                feat[f"L{L}_think_max"] = h[a:b].float().amax(0).cpu().numpy()
                # position-resolved samples (store strided absolute offsets)
                offs = list(range(a, b, THINK_STRIDE))
                feat[f"L{L}_think_pos"] = h[offs].float().cpu().numpy() if offs else None
        rows.append({"id": rec["id"], "feat": feat,
                     "think_len_tok": anchors["think_span"][1] - anchors["think_span"][0]})
        if j % 20 == 0:
            print(f"  [{rec['run']}] {j+1}/{len(recs)} (seq={len(input_ids)})", flush=True)
    np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)
    print(f"saved {len(rows)} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/activations"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = [json.loads(l) for l in open(args.dataset)]
    tok = AutoTokenizer.from_pretrained(BASE)

    for run, adapter in ADAPTERS.items():
        recs = [r for r in ds if r["run"] == run]
        print(f"=== {run}: {len(recs)} rollouts, adapter {adapter} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                     device_map="cuda")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        run(model, tok, recs, args.out_dir / f"act_{run}.npy")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
