#!/usr/bin/env bash
# One-shot GPU-box setup (off-clock infrastructure). Tested target: single 24GB+ GPU (4090/A100).
set -euo pipefail

# --- env ---
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch transformers peft accelerate scikit-learn matplotlib pandas \
  "huggingface_hub[cli]" beautifulsoup4 openai

# --- models (base once + two small LoRA adapters; skip DeepSpeed optimizer shards) ---
hf download allenai/Olmo-3-7B-Instruct-SFT
hf download ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220 \
  --include "adapter_*" --include "*.json" --include "*.jinja" --include "README.md"
hf download ai-safety-institute/cc-olmo3-7b-sutl-b0.02-s720 \
  --include "adapter_*" --include "*.json" --include "*.jinja" --include "README.md"

# --- sanity check: load base + adapter, one forward pass ---
python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "allenai/Olmo-3-7B-Instruct-SFT"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220")
x = tok("def solve():", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**x, output_hidden_states=True)
print(f"OK: {len(out.hidden_states)} hidden states, d_model={out.hidden_states[0].shape[-1]}")
EOF

echo "Setup complete."
