# resi-doom

**You walk inside a language model while it generates. It looks like Doom.**

**[▶ open it](https://unt1l1f1nd-resi-doom.static.hf.space)**

![walking through the layers while the model thinks](docs/walkthrough.gif)

*(smoother: [docs/walkthrough.mp4](docs/walkthrough.mp4))*

29 chambers, one per layer of Qwen3-0.6B. It is already thinking when you walk
in. `W` and `S` walk. That is the control scheme.

- **sixteen windows a chamber** — one per attention head, each showing that
  head's attention matrix, with a line running down it as the model writes
- **blue numbers on the wall** — the residual stream: the twelve largest
  components at that layer, with their indices
- **orange numbers under them** — the loudest neurons of that layer's MLP
- **over the door** — the word this layer would say now, with the runner-up
  beside it when the two are close, and under it `ATTN 217  MLP 380`: the
  length of the vector each half of the block just added to the stream. Read
  it as *which half wrote more here*, and nothing else — a big addition is not
  a big effect, and with a couple of dimensions carrying most of the norm it
  often means "the MLP wrote a lot into `#2`" rather than "the MLP did a lot".
  Layers cannot be compared this way either: both halves grow with depth
  regardless.
- **how dark a chamber is** — how much of its attention sits on the first token

Two things fell out of the measurement rather than the design.

**The answer does not fade in, it switches on.** On Qwen2.5-0.5B asked for the
capital of France, layers 1–21 read as broken C++ and Chinese fragments and
layer 22 reads `' Paris'` at p = 0.61.

**And the two models handle the first token oppositely.** In the tunnel above
(Qwen3-0.6B) the attention sitting on token 0 *climbs* with depth — 0.70 at
layer 3, 0.76 at layer 27 — so the corridor darkens as you go and never
recovers. On Qwen2.5-0.5B it runs at up to 0.84 through the middle and then
**collapses to 0.01 at layer 22**, the same layer where the answer appears.
That is the *attention sink* ([Xiao et al. 2023](https://arxiv.org/abs/2309.17453)),
and token 0 is a control token carrying no content.

Three ways in: the **recorded** run above · **your own sentence**, with
SmolLM2-135M running in your tab and its windows filling in row by row as it
writes · **your own model**, live, through
[brainscope](https://github.com/moudrkat/brainscope).

`IDDQD` `IDKFA` `IDCLIP` `IDDT` work.

`?auto=6` walks by itself · `?hq=1` renders crisp instead of chunky ·
`?rec=40` records the tab for forty seconds and hands you a webm — the whole
tab, bar and captions included, so Chrome asks once to share it; click Share.
Keep that window in front while it runs — browsers throttle rendering in a
window you are not looking at, and you get one frame a second instead of thirty.

<details><summary>What is measured and what is not</summary>

The floor plan is invented — a corridor had to be some shape. Every window,
light, sign and number comes from one forward pass over a real generation
(`tools/compile_wad.py`, on a GPU).

The **logit lens is an approximation**: it pushes mid-stack states through a
final norm and unembedding that were never meant for them, so the babbling
middle may be the instrument rather than the model
([tuned lens](https://arxiv.org/abs/2303.08112)). One prompt, one model,
greedy decoding.

In-browser mode is 4-bit except the readout, which stays exact: that costs
about three of thirty-one door signs against full precision. brainscope mode
has no matrices or neurons in its stream, so those stay blank.

```bash
python tools/compile_wad.py Qwen/Qwen3-0.6B 220 wad.json   # needs a GPU
python -m http.server 8778
```
</details>
