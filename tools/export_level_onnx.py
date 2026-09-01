"""Export a small LM as a graph that emits level data — now including the
attention row per head, so the browser can build the matrix up as it writes."""
import os, sys, glob, torch, torch.nn as nn, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = sys.argv[1]; TAG = sys.argv[2]
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
        attn_implementation="eager").eval()
cfg = model.config
print(f"{MID}: L={cfg.num_hidden_layers} H={cfg.num_attention_heads} "
      f"hidden={cfg.hidden_size} vocab={cfg.vocab_size}")

class Level(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m=m; s.norm=m.model.norm; s.head=m.lm_head
        # the readout stays a Linear on purpose: MatMulNBits leaves a Gemm
        # alone, and 4-bit on this matrix drops lens agreement 90% -> 58%.
        # It is the matrix we are reading; it does not get compressed.
    def forward(s, input_ids):
        o = s.m(input_ids=input_ids, output_attentions=True,
                output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[:, -1, :] for h in o.hidden_states], 0)[:, 0]
        # HF applies the final norm before appending the last hidden state, so
        # the last row is already normed - norming it again is what broke the
        # readout on the top floor.
        z = torch.cat([s.head(s.norm(hs[:-1])), s.head(hs[-1:])], 0)
        logits = z[-1:]
        p = z.softmax(-1)
        lens_p, lens_id = p.max(-1)
        a = torch.stack([x[0, :, -1, :] for x in o.attentions], 0)     # (L, H, T)
        return (logits, lens_id.to(torch.int32), lens_p,
                a[:, :, 0], -(a * (a + 1e-9).log()).sum(-1), a)

wrap = Level(model).eval()
ex = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad(): ref = wrap(ex)
for n, t in zip(["logits","lens_id","lens_p","sink","ent","attn"], ref):
    print(f"  {n:8s} {tuple(t.shape)}")

raw = f"{TAG}.onnx"
torch.onnx.export(wrap, (ex,), raw, opset_version=18, dynamo=True,
    input_names=["input_ids"],
    output_names=["logits","lens_id","lens_p","sink","ent","attn"],
    dynamic_axes={"input_ids":{1:"seq"}})

import onnx, onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig)
q = MatMulNBitsQuantizer(onnx.load(raw),
    algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=False, bits=4))
q.process(); q.model.save_model_to_file(f"{TAG}.q4.onnx", use_external_data_format=True)

def size(p): return (os.path.getsize(p)+sum(os.path.getsize(f) for f in glob.glob(p+"*data")))/1e6
base = ort.InferenceSession(raw).run(None, {"input_ids": ex.numpy().astype(np.int64)})
out  = ort.InferenceSession(f"{TAG}.q4.onnx").run(None, {"input_ids": ex.numpy().astype(np.int64)})
print(f"\n  fp32 {size(raw):7.1f} MB   next={tok.decode([int(base[0][0].argmax())])!r}")
print(f"  4bit {size(TAG+'.q4.onnx'):7.1f} MB   next={tok.decode([int(out[0][0].argmax())])!r}"
      f"   lens {(out[1]==base[1]).mean()*100:.0f}%")
