"""Compile a thinking model's whole generation into one Doom level file."""
import json, math, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MID   = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
NGEN  = int(sys.argv[2]) if len(sys.argv) > 2 else 220
OUT   = sys.argv[3] if len(sys.argv) > 3 else "wad.json"
PROMPT = ("A raindrop lands on a window that is already wet and disappears. "
          "Why? Think it through, then answer.")

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.float32, attn_implementation="eager").cuda().eval()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
print(f"MODEL {MID}: layers={L} heads={H} hidden={cfg.hidden_size}")

msgs = [{"role": "user", "content": PROMPT}]
try:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
except TypeError:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.cuda()
n_prompt = ids.shape[1]

# 1) generate normally (fast, with cache)
with torch.no_grad():
    full = model.generate(ids, max_new_tokens=NGEN, do_sample=False,
                          pad_token_id=tok.eos_token_id)
gen_ids = full[0, n_prompt:].tolist()
print(f"prompt {n_prompt} tok, generated {len(gen_ids)} tok")
print("---\n" + tok.decode(gen_ids, skip_special_tokens=False)[:900] + "\n---")

grab = {}
def cap(name):
    def h(mod, args, out):
        grab[name] = out[0] if isinstance(out, tuple) else out
    return h
for l, blk in enumerate(model.model.layers):
    blk.self_attn.register_forward_hook(cap(f"a{l}"))
    blk.mlp.register_forward_hook(cap(f"m{l}"))
    def pre(name):
        def h(mod, args): grab[name] = args[0]
        return h
    blk.mlp.down_proj.register_forward_pre_hook(pre(f"n{l}"))
    # what each head WROTE, before o_proj mixes the heads: (T, H*head_dim)
    blk.self_attn.o_proj.register_forward_pre_hook(pre(f"h{l}"))

# 2) ONE teacher-forced pass over the finished sequence gives every internal
#    state we need — identical to greedy decoding, 200x cheaper.
norm, head = model.model.norm, model.lm_head
with torch.no_grad():
    o = model(input_ids=full, output_attentions=True, output_hidden_states=True)

import base64
steps, FULL_R, FULL_M, RSCALE, MSCALE, TOPS = [], [], [], [], [], []
FULL_A, FULL_O, FULL_H, ASCALE, OSCALE, HSCALE = [], [], [], [], [], []
for t in range(len(gen_ids)):
    q = n_prompt + t - 1                    # position that predicted token t
    # top three, not one: mid-stack the runners-up are often a hair behind,
    # and that near-tie is the whole character of the readout
    floors = []
    for l in range(L + 1):
        z = head(norm(o.hidden_states[l][0, q]))
        p = z.softmax(-1); top = p.topk(3)
        floors.append([[tok.decode([i]), round(v, 3)]
                       for i, v in zip(top.indices.tolist(),
                                       [round(float(x), 3) for x in top.values])])
        if l == L:   # the last readout, wider: what the whole vocabulary looked like
            tk = p.topk(200)
            TOPS.append([[int(i), round(float(v), 4)] for i, v in zip(tk.indices.tolist(), tk.values.tolist())])
    # the twelve largest components of the residual stream at each layer,
    # with their indices — these run along the walls
    # what each half of the block actually contributed at this position
    add = []
    for l in range(L):
        add.append([round(float(grab[f"a{l}"][0, q].norm()), 1),
                    round(float(grab[f"m{l}"][0, q].norm()), 1)])
    # the eight loudest neurons in each MLP, with their indices
    neu = []
    for l in range(L):
        v = grab[f"n{l}"][0, q]
        _, i = v.abs().topk(8)
        neu.append([[int(j), round(float(v[j]), 1)] for j in i.tolist()])
    res, strip = [], []
    for l in range(L + 1):
        h = o.hidden_states[l][0, q]
        v, i = h.abs().topk(12)
        res.append([[int(j), round(float(h[j]), 1)] for j in i.tolist()])
        # the whole vector, int8 with one scale per (step, layer) — the floor
        # you walk on is every dimension, not twelve of them
        sc = float(h.abs().max()) / 127 or 1.0
        FULL_R.append((h / sc).round().clamp(-127, 127).to(torch.int8).cpu().numpy().tobytes())
        RSCALE.append(round(sc, 6))
        # 64 buckets of max |h|, kept for the older readers of the strip
        b = h.abs().reshape(64, -1).amax(-1); b = b / b.max().clamp_min(1e-9)
        strip.append(base64.b64encode((b * 255).round().to(torch.uint8).cpu().numpy().tobytes()).decode())
    def q8(v, store, scales):
        sc = float(v.abs().max()) / 127 or 1.0
        store.append((v / sc).round().clamp(-127, 127).to(torch.int8).cpu().numpy().tobytes())
        scales.append(round(sc, 6))
    for l in range(L):
        q8(grab[f"n{l}"][0, q], FULL_M, MSCALE)     # every MLP neuron (intermediate)
        q8(grab[f"a{l}"][0, q], FULL_A, ASCALE)     # what the attention block added to h
        q8(grab[f"m{l}"][0, q], FULL_O, OSCALE)     # what the MLP block added to h
        q8(grab[f"h{l}"][0, q], FULL_H, HSCALE)     # per head, what it wrote (H x head_dim)
    lights = []
    for l in range(L):
        a = o.attentions[l][0, :, q, :q+1]
        pr = a.clamp_min(1e-9)
        ent = (-(pr*pr.log()).sum(-1) / math.log(max(2, q+1)))
        # how far back each head is looking, in tokens (0 = itself)
        back = (q - a.argmax(-1)).tolist()
        lights.append([[round(x,2) for x in ent.tolist()],
                       [round(x,2) for x in a[:,0].tolist()],
                       [int(b) for b in back]])
    steps.append({"t": tok.decode([gen_ids[t]]), "f": floors, "l": lights, "r": res, "a": add, "n": neu, "p": strip})

# ---- one stained-glass window per head: its whole attention matrix ----
# max-pooled to 32x32 (max, not mean: it keeps the thin diagonals and the
# first-column stripe that make a head recognisable), stored as uint8 base64.
import base64, torch.nn.functional as F
G = 32
panes = []
for l in range(L):
    a = o.attentions[l][0]                       # (H, T, T) over the whole run
    T = a.shape[-1]
    pad = (-T) % G
    ap = F.pad(a, (0, pad, 0, pad))
    small = ap.reshape(a.shape[0], (T+pad)//G, G, (T+pad)//G, G).amax(dim=(1, 3)) \
        if False else F.adaptive_max_pool2d(ap.unsqueeze(1), (G, G)).squeeze(1)
    small = small / small.amax(dim=(1, 2), keepdim=True).clamp_min(1e-9)
    q = (small.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    panes.append([base64.b64encode(q[h].tobytes()).decode() for h in range(H)])
print(f"\npanes: {L} layers x {H} heads x {G}x{G}")

# ---- everything: every residual dimension and every MLP neuron, every step ----
# int8 with one float scale per (step, layer); a flat sidecar, not JSON, so the
# level file stays small. Layout: all residual vectors (step-major, L+1 per
# step), then all MLP vectors (step-major, L per step).
import os
BIN = os.path.splitext(OUT)[0] + ".bin"
# step-major blocks, in this order: h (L+1 x hidden), mlp neurons (L x inter),
# attn added (L x hidden), mlp added (L x hidden), per-head writes (L x H*head_dim)
with open(BIN, "wb") as f:
    for store in (FULL_R, FULL_M, FULL_A, FULL_O, FULL_H):
        for b in store: f.write(b)
D_H, D_M = cfg.hidden_size, cfg.intermediate_size
D_HD = len(FULL_H[0]) if FULL_H else 0
S = len(steps)
full = {"file": os.path.basename(BIN), "dtype": "int8",
        "hidden": D_H, "inter": D_M, "headwrite": D_HD,
        "rscale": [RSCALE[i*(L+1):(i+1)*(L+1)] for i in range(S)],
        "mscale": [MSCALE[i*L:(i+1)*L] for i in range(S)],
        "ascale": [ASCALE[i*L:(i+1)*L] for i in range(S)],
        "oscale": [OSCALE[i*L:(i+1)*L] for i in range(S)],
        "hscale": [HSCALE[i*L:(i+1)*L] for i in range(S)]}
print(f"\n{BIN}: {os.path.getsize(BIN)/1e6:.2f} MB — {S} steps x ({L+1}x{D_H} h + {L}x{D_M} mlp + 2x{L}x{D_H} added + {L}x{D_HD} head writes) int8")

# every word the model could have said: the whole vocabulary, once, plus the
# top-200 of the final readout at every step (id, p) — the wall at the end
vocab = [tok.decode([i]) for i in range(len(tok))]
for st, tp in zip(steps, TOPS): st["top"] = tp
wad = {"model": MID, "pane": G, "panes": panes, "n_prompt": n_prompt, "layers": L, "heads": H, "prompt": PROMPT,
       "said": tok.decode(gen_ids, skip_special_tokens=False), "steps": steps, "full": full,
       "vocab": vocab}
json.dump(wad, open(OUT, "w"), ensure_ascii=False, separators=(',',':'))
print(f"\n{OUT}: {os.path.getsize(OUT)/1e6:.2f} MB, {len(steps)} steps")

# where the answer surfaces, per step — the level's dramaturgy
print("\n== first floor whose top-1 equals the emitted token ==")
for t in range(0, len(steps), max(1, len(steps)//24)):
    s = steps[t]; want = s["t"]
    hit = next((l for l, fe in enumerate(s["f"]) if fe[0][0] == want), None)
    print(f"  step {t:>3} {want!r:<12} surfaces at floor {hit}")
sink = [sum(sum(st['l'][l][1])/H for st in steps)/len(steps) for l in range(L)]
print("\n== loudest MLP neurons (step 0) ==")
for l in range(0, L, 4):
    print(f"  layer {l:>2}: " + "  ".join(f"#{d}:{v:.0f}" for d, v in steps[0]["n"][l][:5]))
print("\n== what each half of the block adds (step 0: |attn| / |mlp|) ==")
for l in range(0, L, 3):
    a_, m_ = steps[0]["a"][l]
    print(f"  layer {l:>2}: attn {a_:>8.1f}   mlp {m_:>8.1f}   mlp/attn {m_/max(a_,1e-9):>6.2f}")
print("\n== how close are the runners-up? (step 0) ==")
for l in range(0, L+1, 4):
    f0 = steps[0]["f"][l]
    print(f"  layer {l:>2}: " + "   ".join(f"{w!r}:{pv:.2f}" for w, pv in f0))
print("\n== how far back the heads look, per floor (step 0: median / max tokens) ==")
import statistics
for l in range(0, L, 3):
    b = steps[0]["l"][l][2]
    print(f"  floor {l:>2}: median {statistics.median(b):>6.0f}   max {max(b):>5}   "
          f"local heads (<=3 back): {sum(1 for x in b if x<=3)}/{H}")
print("\n== biggest residual component per floor (step 0) ==")
for l in range(0, L+1, 3):
    top = steps[0]["r"][l][0]
    print(f"  floor {l:>2}: dim {top[0]:>5} = {top[1]:>10.1f}   "
          + " ".join(f"{d}:{v:.0f}" for d, v in steps[0]["r"][l][1:5]))
print("\n== mean sink per floor (over all steps) ==")
print("  " + "  ".join(f"{l}:{v:.2f}" for l, v in enumerate(sink)))
