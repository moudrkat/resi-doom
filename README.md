# qwen-doom — E1M1

**You walk around inside a language model. It looks like Doom.**

A static page: a tunnel of **29 chambers, one per layer** of
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). You walk forward, you
read what the model is thinking at that depth, and you press **GENERUJ** to
make it think — the whole tunnel then rewrites itself around you, one token
at a time.

Two controls. `W` forward, `S` back. The mouse only looks.

## What is measured and what is invented

Everything that lights up, glows or is written on a wall comes from one
forward pass over a real generation (`tools/compile_wad.py`, run on a GPU,
220 tokens, thinking enabled). **The floor plan is not measured** — a corridor
had to be *some* shape and I chose it. These are:

| in the tunnel | in the model |
|---|---|
| the sign over each door | logit lens — the top-1 token that layer would emit *now* |
| how bright a chamber is | how much of its attention goes into the first token, averaged over the run |
| red beams pointing backwards | heads whose attention mass sits on token 0 (>30 %) |
| lamps on the walls | one per attention head; brightness = 1 − entropy of its attention |
| the blue pipe overhead | the residual stream, running through every layer |
| the subtitle | the tokens as they are emitted, thinking included |

## The two findings the level is built around

Measured on Qwen2.5-0.5B (`What is the capital of France?`) and confirmed in
shape on Qwen3-0.6B:

1. **The answer does not crystallise gradually.** Layers 1–21 read as broken
   C++ and Chinese fragments (`")));`, `/*****`, `换句话`, `;break`). Layer 22
   reads `' Paris'` at p = 0.613. It is a switch, not a fade.
2. **Layers 3–21 pour up to 84 % of their attention into the first token**,
   which carries no meaning — the *attention sink*
   ([Xiao et al. 2023](https://arxiv.org/abs/2309.17453)). At layer 22 that
   drops to ~1 %. Two independent measurements, the same boundary.

## Honest caveats (they are also sprayed on the walls)

- The logit lens is an **approximation** — it pushes mid-stack states through
  a final norm + unembedding that were never meant for them. The babbling
  middle floors may be the *instrument*, not the model. A tuned lens
  ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112)) is the fix.
- One prompt, one model, greedy decoding. Nothing here is a statistic.

## Rebuild the level

```bash
python tools/compile_wad.py Qwen/Qwen3-0.6B 220 wad.json   # needs a GPU
python -m http.server 8778                                  # then open index.html
```

## Three ways in

- **RECORDED** — Qwen3-0.6B, 220 tokens with thinking on, compiled ahead of
  time. 1 MB, instant, nothing to download.
- **YOUR OWN SENTENCE** — SmolLM2-135M runs in your tab (177 MB, once).
  The published ONNX exports were no use: they emit only `logits` and
  `present.N.key/value` — no hidden states, no attentions — so `tools/`
  exports a graph that **emits the level itself**: logit-lens argmax per
  layer, per-head sink and attention entropy. A token costs a few hundred
  numbers instead of megabytes. Verified against PyTorch: `lens_id` exact,
  floats to 3e-5.
- **CONNECT BRAINSCOPE** — point it at your own
  [brainscope](https://github.com/moudrkat/brainscope) server and the same
  tunnel runs on your own model. Its websocket already streams `lens` and
  `head_entropy` per layer per token; per-head sink is not in that payload,
  so the beams there fall back to `attn_top == 0`. **Untested against a live
  server** — the recorded and in-browser paths are the ones I have run.

### Cheats

`IDDQD` turns the sector lighting off — which is the same as turning the
attention sink off, because the sink is what made the middle of the tunnel
dark. `IDKFA` puts the probability on every sign and lights every head's
beam, not just the ones over 30 %. `IDCLIP` lets you leave the tunnel and
look at the network from outside. `IDDT` looks down.

`?auto=1` walks forward on its own — for recording. `?f=12` is not a thing;
walk there.

## One number you should know before trusting the live mode

The in-browser model is 4-bit quantised, and that is not free
(measured on `The capital of France is`, one prompt):

| build | size | signs identical to fp32 | emitted token |
|---|---|---|---|
| fp32 | 545 MB | 31/31 | ` Paris` |
| 8-bit MatMulNBits | 240 MB | 31/31 | ` Paris` |
| **4-bit, block 32** | **177 MB** | **28/31** | ` Paris` |
| 4-bit, block 64 | 177 MB | 21/31 | ` the` |

8-bit is the honest build but `onnxruntime-web` would not run it — its
`MatMulNBits` kernel took the 4-bit weights and refused the 8-bit ones. So
the tab runs 4-bit, and about three of the thirty-one signs differ from what
the full-precision model would have written. The recorded level has no such
caveat: it comes straight from PyTorch.
