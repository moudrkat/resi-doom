# resi-doom

**You walk inside a language model while it generates. It looks like Doom.**

**[▶ open it](https://unt1l1f1nd-resi-doom.static.hf.space)**

![walking through the layers while the model thinks](docs/walkthrough.gif)

*(smoother: [docs/walkthrough.mp4](docs/walkthrough.mp4))*

30 chambers, one per layer of SmolLM2-135M — the same model that runs in
your tab when you type your own sentence. It is already thinking when you walk
in. `W` and `S` walk. That is the control scheme. What is behind the door at
the end of the corridor is yours to find.

- **sixteen windows a chamber** — one per attention head, each showing that
  head's attention matrix, with a line running down it as the model writes
- **the floor** — the residual stream, all of it: every one of the 1024
  dimensions at that layer, blue above zero, orange below, one cell each;
  the twelve largest follow in numbers, with their indices
- **the strip under the windows** — what the attention block added to the
  stream at this step, every dimension
- **the strip under each window** — what that head wrote, before the heads
  are mixed together: its 128 values
- **the wall at the tail** — the MLP, every one of its 3072 neurons after the
  gate, and under them what the block added to the stream; the eight loudest
  neurons follow in numbers
- **over the door** — the word this layer would say now, with the runner-up
  beside it when the two are close, and under it `ATTN 217  MLP 380`: the
  length of the vector each half of the block just added to the stream. Read
  it as *which half wrote more here*, and nothing else — a big addition is not
  a big effect, and with a couple of dimensions carrying most of the norm it
  often means "the MLP wrote a lot into `#2`" rather than "the MLP did a lot".
  Layers cannot be compared this way either: both halves grow with depth
  regardless.
- **how dark a chamber is** — how much of its attention sits on the first token

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
(`tools/compile_wad.py`, on a GPU). The level file is `wad.json` (signs,
windows, lights) plus `wad.bin`, the everything file: every residual
dimension, every MLP neuron, what each block added and what each head wrote,
at every step, as int8 with one scale per vector (~50 MB for 220 tokens).

The **logit lens is an approximation**: it pushes mid-stack states through a
final norm and unembedding that were never meant for them, so the babbling
middle may be the instrument rather than the model
([tuned lens](https://arxiv.org/abs/2303.08112)). One prompt, one model,
greedy decoding.

In-browser mode is 4-bit except the readout, which stays exact: that costs
about three of thirty-one door signs against full precision. brainscope mode
has no matrices or neurons in its stream, so those stay blank.

```bash
python tools/compile_wad.py HuggingFaceTB/SmolLM2-135M-Instruct 220 wad.json   # needs a GPU
python -m http.server 8778
```
</details>
