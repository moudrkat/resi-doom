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
    lights = []
    for l in range(L):
        a = o.attentions[l][0, :, q, :q+1]
        pr = a.clamp_min(1e-9)
        ent = (-(pr*pr.log()).sum(-1) / math.log(max(2, q+1)))
        lights.append([[round(x,2) for x in ent.tolist()],
                       [round(x,2) for x in a[:,0].tolist()]])
    steps.append({"t": tok.decode([gen_ids[t]]), "f": floors, "l": lights})

wad = {"model": MID, "layers": L, "heads": H, "prompt": PROMPT,
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
print("\n== mean sink per floor (over all steps) ==")
print("  " + "  ".join(f"{l}:{v:.2f}" for l, v in enumerate(sink)))
