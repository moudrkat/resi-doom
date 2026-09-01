"""Export a small LM as an ONNX graph that emits LEVEL DATA, not hidden states.

Outputs per forward pass (last position only):
  logits   (1, V)     - for greedy/sampled decoding in JS
  lens_id  (L+1,)     - logit-lens argmax per layer  -> the sign over each door
  lens_p   (L+1,)     - its probability
  sink     (L, H)     - attention mass on token 0    -> the red beams
  ent      (L, H)     - normalised attention entropy -> lamp brightness
"""
import os, sys, math, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "level.onnx"

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.float32, attn_implementation="eager").eval()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
print(f"{MID}: layers={L} heads={H} hidden={cfg.hidden_size} vocab={cfg.vocab_size}")

class Level(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        self.norm = m.model.norm; self.head = m.lm_head
    def forward(self, input_ids):
        o = self.m(input_ids=input_ids, output_attentions=True,
                   output_hidden_states=True, use_cache=False)
        logits = o.logits[:, -1, :]                          # (1, V)
        hs = torch.stack([h[:, -1, :] for h in o.hidden_states], 0)[:, 0]   # (L+1, hidden)
        z = self.head(self.norm(hs))                          # (L+1, V)
        p = z.softmax(-1)
        lens_p, lens_id = p.max(-1)
        a = torch.stack([x[0, :, -1, :] for x in o.attentions], 0)          # (L, H, T)
        sink = a[:, :, 0]
        # raw entropy in nats; JS divides by log(seq) so the graph stays
        # length-agnostic instead of baking the export-time sequence in
        ent = -(a * (a + 1e-9).log()).sum(-1)
        return logits, lens_id.to(torch.int32), lens_p, sink, ent

wrap = Level(model).eval()
ex = tok("Hello there, how are you", return_tensors="pt").input_ids
with torch.no_grad():
    outs = wrap(ex)
for n, t in zip(["logits","lens_id","lens_p","sink","ent"], outs):
    print(f"  {n:8s} {tuple(t.shape)} {t.dtype}")
print("  lens words:", [tok.decode([i]) for i in outs[1].tolist()][:6], "...",
      [tok.decode([i]) for i in outs[1].tolist()][-3:])

torch.onnx.export(
    wrap, (ex,), OUT, opset_version=18, dynamo=True,
    input_names=["input_ids"],
    output_names=["logits","lens_id","lens_p","sink","ent"],
    dynamic_axes={"input_ids":{1:"seq"}, "sink":{2:"seq"}, "ent":{2:"seq"}},
)
print(f"\nwrote {OUT}: {os.path.getsize(OUT)/1e6:.1f} MB")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q = OUT.replace(".onnx", ".q8.onnx")
    quantize_dynamic(OUT, q, weight_type=QuantType.QInt8)
    print(f"wrote {q}: {os.path.getsize(q)/1e6:.1f} MB")
except Exception as e:
    print("quantisation unavailable:", type(e).__name__, e)

# ---------------------------------------------------------------------------
# Shipping notes (measured 2026-09-01, SmolLM2-135M-Instruct):
#   fp32                     545 MB   lens 100%   next=' Paris'   (reference)
#   MatMulNBits 8bit blk32   240 MB   lens 100%   next=' Paris'   <- shipped
#   MatMulNBits 4bit blk32   185 MB   lens  90%   next=' Paris'
#   MatMulNBits 4bit blk64   177 MB   lens  68%   next=' the'     <- breaks it
#   fp16                     conversion fails: keep_io_types mismatch on _to_copy
# 4-bit changes the very numbers the level claims to measure, so it is not used.
#
#   uv pip install --python <venv>/bin/python --target ./pylibs onnxscript onnx onnxruntime
#   PYTHONPATH=./pylibs python tools/export_level_onnx.py HuggingFaceTB/SmolLM2-135M-Instruct level.onnx
