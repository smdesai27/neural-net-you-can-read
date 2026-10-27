# A neural net you can read

Working title. Spec from 2026-10-26.

**Status, 2026-10-26.** Steps 1–8 are built and tested (15 tests; the page has been checked
in headless Chrome at desktop and phone widths). Step 9, publishing, is not done. Changes
from the plan while building:
- The fork has 8,470 built operations; 6,693 of them feed an output, in 201 layers. The
  minimum-width change deepened the circuit from upstream's 147 layers.
- Lesson 2 no longer says the network is piecewise linear. Its neurons include ReGLUs (a
  value times a ReLU), so between seams it is one fixed smooth formula, not a linear one.
- Lesson 3 tests the 553 neurons that sit behind a decision (every path from them to a tile
  crosses a 0/1 step). All 553 have gradient exactly 0, and knocking out about 130 of them
  still moves tiles.
- Lesson 5 defaults to probing the page width. It is the one target where a layer reads the
  fact perfectly (R² 1.00) and erasing it there moves nothing, for every layout tried.
- Breeding rejects invalid children instead of repairing them. 150–220 of every 256 tries
  are valid, which is enough.

A standalone demo site, not a homepage. A page layout is computed by a neural network:
8,385 neurons in 147 layers, every weight written by hand with
[handwired](https://github.com/chenglou/handwired). You can breed the layout like
Picbreeder. Then a short interpretability lesson uses the layouts you just bred to teach
reading neurons, ablation, activation patching and probing. Since a human wrote every
neuron, each method can be checked against an answer key.

## The page

One scrolling page in two parts. A stage stays pinned throughout: the live layout on top,
the network underneath it, with 147 layers drawn as columns and one dot per neuron, colored
by activation.

### Part 1: play

- **Stage.** Twelve photo tiles laid out by the network. Drag the stage width and the
  layout re-flows while activations ripple through the columns.
- **Breed.** A 3×3 grid of children. Pick one or two parents and breed. Every pick is a
  vote ("this over those"), and the votes train a 7-weight taste model, so later
  generations show what it predicts you'll like, plus 2 wildcards. Any layout can be shared
  as a link.
- **Atlas.** A map of layout space. Across: stage width, 360–1920 px. Down: one gene you
  choose. Lines mark where the layout changes. Hover for "next change at 1,104 px: row 2
  goes from 3 to 4 tiles". Click to adopt.

### Part 2: look inside

Every lesson uses your own bred layouts as its examples. Each one has the same shape: what
you do, what you see, what it means for real models, and the caveat.

**1. Read a neuron.**
- **Do:** hover a tile to light up the neurons that compute it. Hover a neuron for its
  label ("slot 4: starts a new row?") and the tiles it affects.
- **Lesson:** in a trained model, nobody hands you these labels; interpretability is
  guessing them. Here you have the answer key. That's the idea behind
  [Tracr](https://arxiv.org/abs/2301.05062) and
  [InterpBench](https://arxiv.org/abs/2407.14494), as something you can play with.

**2. Seams are neurons flipping.**
- **Do:** click a line in the atlas. It jumps to the one neuron that crosses zero there,
  and plots that neuron's input against stage width so you can see the ReLU kink.
- **Lesson:** inside a region where no ReLU switches and no decision flips, the network is
  one fixed formula; it changes behavior only at region boundaries. That's the
  ["polytope lens"](https://arxiv.org/abs/2211.12312) view of ReLU nets, made visible.

**3. Break it: ablation vs. gradients.**
- **Do:** zero out neurons and watch what breaks. Then compare with gradient attribution,
  which reports some of those neurons as not mattering at all.
- **Lesson:** gradients only see the local linear region, so they miss effects that cross
  a decision boundary. In the repo's drag experiment, 24.6% of frames had zero gradient on
  every knob. This is why saliency maps mislead and why ablation and integrated gradients
  exist.

**4. Patch it.**
- **Do:** take two of your bred layouts. Copy the second one's activations into the first,
  one layer at a time, and plot how much of the second layout appears.
- **Lesson:** activation patching shows where a decision is computed. The labels tell you
  whether patching found the right layers. Where several paths carry the same information,
  the curve smears, the same trap as "backup heads" in real circuits work.

**5. Probe it: decodable isn't used.**
- **Do:** train a linear probe at every layer to read a fact (slot 3's row, slot 7's
  importance) from activations collected while you bred. Then delete the probe's direction
  and see whether the layout changes.
- **Lesson:** high probe accuracy doesn't mean the network uses the information. In the
  layers after a fact is consumed, it stays decodable, but erasing it changes nothing. See
  [control tasks](https://arxiv.org/abs/1909.03368) and
  [amnesic probing](https://arxiv.org/abs/2006.00995).

**Close: what real models don't give you.**
- This network is clean, sparse and labeled, with no
  [superposition](https://transformer-circuits.pub/2022/toy_model/index.html). Trained
  networks aren't like that.
- The point isn't that real models look like this. It's that here you can check whether a
  method gives the right answer.
- Teaser for a later part: unfreeze and mutate the weights, then watch the labels stop
  being true.

## Under the hood

**Circuit.** Fork of `lab/generative-fields/flow-layout.ts`. It's 8,385 operations and 147
layers, and one evaluation takes about 60 µs on the CPU. It places 16 slots by greedy row
wrapping and checks its own result for overlap, overflow and minimum size, exactly.

There's one change. Tile widths are currently fractions of the page width, so the columns
are the same at every width. Add a minimum tile width in px:
`span = max(fraction, minTilePx × invInnerWidth)`. The host supplies `invInnerWidth` as an
input cell, following the repo's rule that division stays outside the circuit. Columns then
change with width, which gives the atlas its seams. The float64 oracle (`oracle.ts`) gets
the same change, and the tests compare the two again.

**Labels.** The forked builder tags wires as it creates them: per slot (span, height, rank
bits, wrap decision, cursor, row, final box) and per check (overlap pair, containment). An
untagged operation takes the label of the nearest tagged wire it feeds, for example
"part of slot 3's wrap decision."

**Instrumented evaluator.** It walks `circuit.operations` in order with an overrides map
(zero, mean, or a value copied from another run) and returns all 8,385 values. With no
overrides it must match `circuit.evaluateReference` bit for bit; a test checks that.
Gradients come from the runtime's existing `backpropagateReference`.

**Network view.** One canvas: 147 columns, up to 497 dots each. Dots are colored by
activation, by label group, or by patch effect.

**Probes.** Logistic or linear regression in JS, per layer, on activations from the
session's bred layouts and atlas tiles (thousands of samples). Amnesic step: project out
the probe direction, re-run, and compare.

**Breeding and taste.**
- Genome: 8 global knobs, 4 numbers per visible slot, and `minTilePx`. Visibility is
  pinned on for the 12 tiles.
- Mutation: Gaussian noise scaled per gene's range, plus a rare large jump.
- Crossover: each gene comes from one parent or the other.
- Each generation makes 256 candidates (about 15 ms on the CPU). Invalid ones are repaired
  with the circuit's own `projectToValidity` or dropped.
- Taste: a 7-weight logistic regression on differences in the experiment's fitness terms.
  The repo lists this as a next step that was never tried.

**Atlas.** 195 widths × 32 gene values is about 6k layouts, roughly 0.4 s on the CPU in a
Web Worker, drawn progressively. The "next change" readout comes from decision margins,
which give exact flip points; the repo's drag experiment got 303 of 303 right.

**Content.** The 12 photos the repo's own demos use (MIT licensed), with short captions.

## What is handwired and what is plain JS

- **Handwired:** the layout network, its validity checks, decision margins, and the repair
  step.
- **Plain JS:** DOM rendering, breeding, the taste model, probes, the instrumented
  evaluator (it walks handwired's operation list), the atlas and network canvases, and URL
  encoding.
- **Stretch:** port the network to WebGPU with the repo's lab WGSL writer, so the atlas
  stays live while you breed.

## Build order

1. **Vendor.** Copy `src/runtime` and `lab/generative-fields` from chenglou/handwired at
   `257856e`, keeping its license. Their tests pass here.
2. **Stage.** Layout plus network view, live width drag.
3. **Minimum tile width.** Circuit and oracle changed, tests pass, columns change with
   width.
4. **Labels and instrumented evaluator.** Matches the reference bit for bit with no
   overrides.
5. **Breed, taste, share links.**
6. **Atlas.**
7. **Lessons 1–4.**
8. **Lesson 5 (probes)** and the closing section.
9. **Ship** as a GitHub Pages project page.
10. **Later:** unfreeze and mutate weights, then measure how labels drift.

## Limits

- This handwritten network isn't a trained one. The lesson says so, and that's the point.
- It's the circuit's layout model, not CSS: up to 16 tiles, one screen.
- A hero tile can't span two rows beside a stacked column.
- The fitness terms can be gamed; pinned visibility and learned taste weights are the guard.
- On the CPU, the atlas takes about half a second to fill in.

## Open decisions

- The name.
- Tile content: the repo's photos (default), or something with more personality.
- Hosting: a GitHub Pages project page under smdesai27 (default).
