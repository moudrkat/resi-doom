# resi-doom · E1M1

**You walk around inside a language model. It looks like Doom.**

![walking through the layers while the model thinks](docs/walkthrough.gif)

**You are the residual stream.** The corridor is the thing that passes through
every layer, and you walk down it. 29 chambers, one per layer of Qwen3-0.6B.
It starts thinking the moment you walk in, and the whole tunnel rewrites itself
around you, token by token. `W` forward, `S` back — that is the control scheme.

- **sixteen windows per chamber** — one per attention head, each showing that
  head's own attention matrix. It is causal, so the lit part is a lower
  triangle. A diagonal means the head reads the previous word; a bright left
  column means it is parked on the first token; slanted stripes mean it is
  copying. **The glass fills in as the model writes**, because the rows of the
  matrix are positions in the text.
- **blue numbers along the wall** — the twelve largest components of the
  residual stream at that layer, with their indices. `#35` grows from 5.8 at
  layer 3 to 102 at layer 27: *massive activations*
  ([Sun et al. 2024](https://arxiv.org/abs/2402.17762)).
- **orange numbers below them** — the eight loudest neurons of that layer's MLP.
- **sign over each door** — the logit lens: the word that layer would say now,
  and under it what each half of the block just added, `ATTN 217  MLP 380`.
- **how dark a chamber is** — how much of its attention sits on the first token,
  a control token that carries no content: the *attention sink*
  ([Xiao et al. 2023](https://arxiv.org/abs/2309.17453)).
- **the map in the status bar** — the whole model, and where you are in it.

## The two things the level is built around

| | |
|---|---|
| **The answer doesn't fade in, it switches on.** | Layers 1–21 read as broken C++ and Chinese fragments. Layer 22 reads `' Paris'` at p = 0.61. |
| **Layers 3–21 pour up to 84 % of their attention into the first token**, which means nothing. | The *attention sink* ([Xiao 2023](https://arxiv.org/abs/2309.17453)). At layer 22 it drops to ~1 %. Two measurements, same boundary. |

## Three ways in

**RECORDED** Qwen3-0.6B, 220 tokens, thinking on · **YOUR OWN SENTENCE**
SmolLM2-135M runs in your tab, 177 MB once · **CONNECT BRAINSCOPE** your own
model, live, through [brainscope](https://github.com/moudrkat/brainscope).

`IDDQD` `IDKFA` `IDCLIP` `IDDT` work. GENERATE pauses and resumes.
`?auto=1` walks by itself, for recording.

<details><summary>What is measured, what I invented, and what the live mode costs</summary>

The floor plan is invented — a corridor had to be some shape. The windows,
the lights, the signs and every number come from one forward pass over a real
generation (`tools/compile_wad.py`, on a GPU).

The **logit lens is an approximation**: it pushes mid-stack states through a
final norm and unembedding that were never meant for them. The babbling middle
of the tunnel may be the instrument, not the model — a
[tuned lens](https://arxiv.org/abs/2303.08112) is the fix. One prompt, one
model, greedy decoding. Nothing here is a statistic.

The published ONNX exports were no use for the browser: they emit only `logits`
and `present.N.key/value`. So `tools/export_level_onnx.py` exports a graph that
**emits the level itself** — lens argmax per layer, per-head sink and entropy —
a few hundred numbers per token instead of megabytes. Verified against PyTorch:
`lens_id` exact, floats to 3e-5.

That graph then has to be quantised, and it is not free (one prompt,
`The capital of France is`):

| build | size | signs identical to fp32 | says |
|---|---|---|---|
| fp32 | 545 MB | 31/31 | ` Paris` |
| 8-bit | 240 MB | 31/31 | ` Paris` |
| **4-bit blk32 — shipped** | **177 MB** | **28/31** | ` Paris` |
| 4-bit blk64 | 177 MB | 21/31 | ` the` |

8-bit is the honest build, but `onnxruntime-web` runs only the 4-bit
`MatMulNBits` kernel. So in your tab about three of thirty-one signs differ
from full precision. The recorded level has no such caveat.

brainscope mode is **untested against a live server**; per-head sink is not in
its websocket payload, so beams there fall back to `attn_top == 0`.

```bash
python tools/compile_wad.py Qwen/Qwen3-0.6B 220 wad.json   # needs a GPU
python -m http.server 8778
```
</details>
