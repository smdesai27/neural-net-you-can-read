# A neural net you can read

A page layout computed by a hand-written neural network (6,693 neurons in 201 layers, built
with [handwired](https://github.com/chenglou/handwired)). You breed the layout, then a
five-part interpretability lesson takes the network apart: reading neurons, seams as
neurons flipping, knockouts vs. gradients, activation patching, and probes. Every neuron was
written by a person, so each method is checked against an answer key.

```sh
bun install
bun start          # dev server for site/index.html
bun test ./tests   # circuit vs. oracle, lens evaluator, breeding, and the lessons' claims
bun run build      # static site in dist/ (relative paths, works under a sub-path)
```

## Layout

- `vendor/handwired/`: handwired's runtime and generative-fields experiment, unchanged (MIT,
  see `VENDOR.md`).
- `src/layout/`: our fork of the layout circuit (a minimum tile width in px, and neuron
  labels), its float64 oracle, the genome schema, and helpers to evaluate a genome.
- `src/lens/`: the network view of the circuit (layers, labels, an evaluator that can
  override any neuron) and the experiments the lessons run.
- `src/breed/`: mutation, crossover, the 7-weight taste model, and share links.
- `site/`: the page.
- `SPEC.md`: what was planned, and what changed while building it.

Deploys to GitHub Pages through `.github/workflows/pages.yml` on a push to `main`.
