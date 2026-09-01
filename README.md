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

## Status

Recorded level: works. **In-browser generation** (type your own sentence) needs
a custom ONNX export — the published exports of Qwen2.5-0.5B and SmolLM2-135M
output only `logits` and `present.N.key/value`, **no hidden states and no
attentions**, so the level cannot be compiled from them. **brainscope connect**
(drive the tunnel live from your own model) is next; its websocket already
streams `lens` and `head_entropy` per layer per token.
