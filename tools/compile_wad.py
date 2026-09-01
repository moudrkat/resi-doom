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

# 2) ONE teacher-forced pass over the finished sequence gives every internal
#    state we need — identical to greedy decoding, 200x cheaper.
norm, head = model.model.norm, model.lm_head
with torch.no_grad():
    o = model(input_ids=full, output_attentions=True, output_hidden_states=True)

steps = []
for t in range(len(gen_ids)):
    q = n_prompt + t - 1                    # position that predicted token t
    floors = []
    for l in range(L + 1):
        z = head(norm(o.hidden_states[l][0, q]))
        p = z.softmax(-1); top = p.topk(1)
        floors.append([tok.decode([top.indices[0].item()]), round(top.values[0].item(), 3)])
    # the twelve largest components of the residual stream at each layer,
    # with their indices — these run along the walls
    # what each half of the block actually contributed at this position
    add = []
    for l in range(L):
        add.append([round(float(grab[f"a{l}"][0, q].norm()), 1),
                    round(float(grab[f"m{l}"][0, q].norm()), 1)])
    res = []
    for l in range(L + 1):
        h = o.hidden_states[l][0, q]
        v, i = h.abs().topk(12)
        res.append([[int(j), round(float(h[j]), 1)] for j in i.tolist()])
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
    steps.append({"t": tok.decode([gen_ids[t]]), "f": floors, "l": lights, "r": res, "a": add})

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

wad = {"model": MID, "pane": G, "panes": panes, "n_prompt": n_prompt, "layers": L, "heads": H, "prompt": PROMPT,
       "said": tok.decode(gen_ids, skip_special_tokens=False), "steps": steps}
json.dump(wad, open(OUT, "w"), ensure_ascii=False, separators=(',',':'))
import os; print(f"\n{OUT}: {os.path.getsize(OUT)/1e6:.2f} MB, {len(steps)} steps")

# where the answer surfaces, per step — the level's dramaturgy
print("\n== first floor whose top-1 equals the emitted token ==")
for t in range(0, len(steps), max(1, len(steps)//24)):
    s = steps[t]; want = s["t"]
    hit = next((l for l,(w,p) in enumerate(s["f"]) if w == want), None)
    print(f"  step {t:>3} {want!r:<12} surfaces at floor {hit}")
sink = [sum(sum(st['l'][l][1])/H for st in steps)/len(steps) for l in range(L)]
print("\n== what each half of the block adds (step 0: |attn| / |mlp|) ==")
for l in range(0, L, 3):
    a_, m_ = steps[0]["a"][l]
    print(f"  layer {l:>2}: attn {a_:>8.1f}   mlp {m_:>8.1f}   mlp/attn {m_/max(a_,1e-9):>6.2f}")
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
