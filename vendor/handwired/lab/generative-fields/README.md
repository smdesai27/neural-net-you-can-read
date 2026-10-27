# Generative layout from fields over one shared slot function

## September 2026 corrections

The tables below are the July study, not promises about the corrected optimizer. The fitness backward pass now includes direct padding and importance derivatives as well as derivatives through geometry. Current gradient-ascent results can therefore differ from the historical numbers. Regression tests check the derivative calculation and useful improvement, not that a particular optimizer continues to produce the old scores.

The historical `validBit` is deliberately unchanged so the old experiment remains interpretable: content may be hidden, horizontal containment allows `innerWidth * wrapSlack` extra pixels, and checks have a 0.5-pixel tolerance. `auditContent(snapshot, required)` adds separate required-content retention and strict padded containment reports, with no wrap allowance or pixel tolerance. A layout can pass the historical bit and fail both stricter reports. Of the July study's 4,154 repairs, 1,408 hid previously visible items; one reduced sixteen visible items to six. Text fit is measured afterward, not part of the optimized score.

The samples do not establish smooth motion: sorting, visibility, and wrapping can jump. The demo may also substitute a nearby saved sample when an intermediate is invalid. Keep sampled validity, content retention, and movement as separate measurements. The new `lab/reusable-layout` experiment keeps required content and importance fixed and allows vertical scrolling.

## Historical July study

One equivariant 16-slot layout circuit, driven entirely by input fields. The same compiled program produces a uniform grid, a hero layout, an importance-weighted mosaic, a list, and a filmstrip purely by changing numbers in the source table; it scores its own validity with exact bits; a Duplo-style fitness function generates layouts by search and by gradient; and straight-line interpolation between valid layouts, repaired by projecting against the circuit's own violation outputs, stays valid at 100% of sampled points (target was 95%).

Reproduce everything: `bun lab/generative-fields/experiments.ts` prints every table below (about 2 seconds); `bun test lab/generative-fields` re-checks the claims (18 tests, about 3 seconds).

## The layout model

Every quantity that makes one slot different from another is a per-slot source cell, never a code branch: `importance` (0..1), `orderKey` (0..1), `visibility` (0..1, hardened at 0.5), `heightPreference` (0..1). Global knobs are source cells too: paddings, gaps, and four coefficients that expand importance into geometry (`baseSpan`, `spanFromImportance`, `baseHeight`, `heightFromImportance`). The generative field vector is these 8 knobs plus 64 per-slot fields = 72 dimensions; the viewport and the validity thresholds are held as context.

The shared per-slot function: each slot turns importance into a width span (fraction of the inner width, `min(1, baseSpan + spanFromImportance * importance)`) and a pixel height (`(baseHeight + heightFromImportance * importance) * (0.5 + heightPreference)`). Slots are ranked by order key — pairwise comparisons in-circuit, ties broken by slot index, so even the tie rule enters as data — and placed by greedy line breaking, the same rule CSS flow and text layout use: a slot that would overflow the current row starts a new row, and a row is as tall as its tallest member. A hero is nothing but an importance peak: a large importance produces a wide span that crowds everything else out of its row.

Design decisions worth recording:

- Rank positions are matched to slots through integer-exact one-hot bits (`clamp01` ramps over the integer rank, expressions.md "Decision bits"), so the routing sums are FP32-lossless: adding fifteen exact zeros to one selected value changes nothing. The boolean atom appears only where a genuinely continuous quantity is thresholded: order key comparisons, visibility hardening, wrap tests, and pixel-slack violation checks — 336 booleans out of 8,385 operations.
- No division anywhere in the circuit. Horizontal quantities stay in fractions of the inner width until one multiply converts them to pixels; every ratio (fill, means, the correlation) lives in the JS scorer, following the repo rule that the parameter/scorer side owns division.
- The order comparison reconstructs the exact FP32 key difference through a pair of relu wires before scaling by 1024, which shrinks the rounding band around the decision threshold to key differences of at most 2^-36, about 1.5e-11 (details under "Failures and limits").
- The one-hot sums routing spans and heights to rank positions carry defensive clamps: if a rank collision inside the comparison band ever routes two slots into one position, the clamped sums keep every downstream wire inside its declared range while the overlap check flags the layout. For legitimate one-hot sums the clamps are exact identities. A test drives the collision with maximal heights and checks every wire of the evaluation against its declared range.
- The sequential flow costs depth, not a fourth atom: 147 layers for 16 slots, each wrap decision feeding the next cursor.

Circuit statistics: 78 source cells, 8,385 operations (4,032 affine, 4,017 reglu, 336 boolean), 147 layers, widest layer 497 outputs, 13,595 stored sparse weights. Build plus tensor compilation ~25ms; one compiled evaluation ~60µs (4,000 evaluations with scoring run in ~230ms).

## Validity is self-scored

The circuit computes, about its own output: pairwise overlap among visible slots (120 pairs), containment (bottom and right edges), minimum visible sizes, each as an exact per-check bit with a 0.5px slack, plus integer violation counts, an aggregate `validBit` (an integer-exact decision bit over the count), and a continuous `violationDepth` in pixels that gradient projection can descend. Everything is verified against a float64 oracle.

A structural result worth stating: the flow parameterization cannot express horizontal overlap or horizontal overflow — the cursor only advances, rows clear their tallest member, and spans are clamped to the row width. The search and interpolation studies track per-category violation counts, and the tests assert them exactly: across the 4,000 search samples, 0 have overlap, 2,886 have vertical overflow, and 44 have minimum-size violations; across all 6,060 interpolation points the overlap check fired exactly zero times. Pushing design from structure into fields makes a whole violation class unreachable, and validity checking concentrates on the constraints that remain.

## Fitness (Duplo-style)

Seven weighted heuristic terms in the spirit of Flipboard's engine: fill (0.25), vertical fit (0.2), importance-area correlation (0.15), horizontal balance (0.1), size comfort (0.1), row evenness (0.05), content richness (0.15). The circuit computes the division-free numerators as wires (covered area, content bottom, violation depths); the JS scorer owns the divisions and carries hand-derived gradients with respect to slot geometry and those numerator wires, verified against central finite differences (300+ checks per run).

The content richness term exists because of a reward-hacking episode: without it, gradient ascent collapsed every start into two or three viewport-filling slabs that maxed out fill, balance, comfort, and evenness simultaneously. Duplo used dozens of terms; even seven leave thin exploits (see gradient results below). The episode is retained as a real test: `scoreFitness` takes an optional weighting, and re-running ascent with the original pre-richness weighting reproduces the collapse exactly — best score 0.998 with 2 visible slots, pinned in `generation.test.ts`.

## Experiment 1 — expressivity: five families, zero recompilation

All five authored field vectors run on the same machine instance; the test asserts the operation count and the compiled tensor program object are unchanged across all evaluations, and asserts each family's structure (grid: 4x4 equal cells; heroRail: the peak slot owns its row and dwarfs the rail; mosaic: 4+ distinct widths over 5+ rows; list: 6 full-width rows, 10 slots hidden with zero geometry; filmstrip: 16 slots in one row).

| family | global knobs (padding, gapX, gapY, span, height) | per-slot fields | valid | fitness |
| --- | --- | --- | --- | --- |
| grid | 24/24, 0.015, 14, span 0.235+0u, height 177+0u | importance 0.2, visibility 1 | yes | 0.894 |
| heroRail | 24/24, 0.015, 12, span 0.2+0.58u, height 70+120u | slot0 importance 0.95, rest 0.12 | yes | 0.863 |
| mosaic | 20/20, 0.012, 12, span 0.18+0.25u, height 48+56u | importance pattern 0.1..0.9, heightPreference pattern 0.3..0.8 | yes | 0.823 |
| list | 40/24, 0.01, 12, span 1+0u, height 110+0u | slots 0..5 visible, rest hidden | yes | 0.826 |
| filmstrip | 24/60, 0.006, 10, span 0.0555+0u, height 640+0u | importance 0.2, visibility 1 | yes | 0.870 |

Order keys are `(slot + 0.5) / 16` everywhere; heroRail with the hero moved to a different slot is just a different importance vector. `experiments.ts` prints the full 72-number vectors and character renders of each family.

## Experiment 2 — generation by search

4,000 seeded field vectors (global knobs from a designer-plausible sub-box, per-slot fields uniform in 0..1), each evaluated through the compiled circuit and scored. 1,078 of 4,000 (26.9%) were valid; among the valid ones:

| statistic | fitness |
| --- | --- |
| min | 0.297 |
| p25 | 0.616 |
| median | 0.688 |
| p75 | 0.745 |
| p90 | 0.792 |
| max | 0.875 |

The counts are exact and reproducible (seeded FP32 end to end); the test pins `validCount == 1078` and re-runs the seed to confirm bit-identical results. Search's winners were layouts I had not authored: the top candidate (fitness 0.875, sample 1731) hides 7 of 16 slots and arranges the rest into a two-hero band over mixed two-and-three column rows; the runner-up (0.870) builds a tall portrait feature column on the left (one slot with importance 0.95 and `heightFromImportance` 193) with small items flowing beside it. The unanticipated part was the visibility field acting as a content-selection knob: search uses hiding as aggressively as sizing. Full field vectors for the top three are in the experiments output.

## Experiment 3 — generation by gradient

Fitness gradients seed the geometry and numerator wires, then flow to the fields through a straight-through backward pass (`ste-backprop.ts`): affine and reglu backpropagate exactly like the repo's reference pass, and the boolean atom passes gradient through unchanged. The forward result stays bit-exact; only the training-time gradient is estimated. Steps are Adam in normalized field space (every field mapped to 0..1 by its declared range), clamped back into range each step.

12 seeded random starts x 150 steps (1,812 evaluations, ~180ms): all 12 end on valid layouts, best 0.922, median 0.887, worst 0.865 (each start's validity and improvement, and these three statistics, are pinned exactly in `generation.test.ts`) — and the best beats search's 0.875 at half the evaluation budget. But the ascended layouts show few slots (the best shows 6, not 16): with the richness term capped at 0.15, ascent still prefers slab-like layouts whose other six terms saturate. Gradient polish of search's top candidate (50 gentle steps) did not improve it (0.875 -> 0.875). Comparison in one sentence: gradient is cheaper and scores higher on the scalar objective, search stays inside the sampling prior and produces the layouts a person would actually pick; the gap is a measurement of fitness under-specification, not of optimizer quality.

The straight-through gradient itself is characterized by test: on purely continuous paths (baseHeight, gapY, paddingY into content bottom) it matches finite differences to within 2%; on fields that also feed discrete decisions (importance feeds the span, the span feeds wrap booleans) it carries phantom gradient through the frozen decisions — measured 173 vs a true local slope of 120 on the hero importance probe — so the test demands sign agreement and a factor-3 magnitude band there.

## Experiment 4 — the headline: interpolation with validity projection

60 pairs of valid layouts (all 10 authored-authored pairs plus 50 seeded pairs from a pool of the 5 families and the top 15 valid searched layouts), 101 points per straight line in the 72-dimensional field space, 6,060 evaluated points:

| metric | value |
| --- | --- |
| valid before projection | 1,906 / 6,060 (31.5%) |
| valid after projection | 6,060 / 6,060 (100.0%) |
| fully valid naive paths | 1 / 60 |
| repaired / failed | 4,154 / 0 |
| mean projection steps | 1.9 (cap 40) |
| mean moved distance (normalized) | 0.208 (max 1.423) |
| per-path naive validity | min 0.02, median 0.22 |

Naive interpolation breaks most paths — every one of the 4,154 invalid points is a vertical overflow (the content bottom lands past the padded viewport bottom); minimum-size violations never appeared along these paths, and overlap is structurally unreachable — and projection repaired every single invalid point: Adam descent on the circuit's own continuous `violationDepth` wire plus a proximal pull toward the interpolated anchor, stopping the moment the exact `validBit` turns on. Repair is fast (mean 1.9 steps) because the violation gradients point straight at the responsible fields: heights, gaps, spans. The moved distance measures how far the repair strays from the straight line: 0.208 over 72 normalized dimensions is roughly a 2.4% nudge per field. The target was 95% valid after projection; measured 100%. The test pins the exact counts: 1,906 naive-valid, 4,154 repaired, 0 failed, 6,060 total, and the per-category breakdown (0 overlap, 4,154 containment, 0 minimum-size).

Hero migration: importance mass moving from slot 2 to slot 11 (identical fields otherwise). All 21 naive points are already valid — no projection needed — and the areas transfer monotonically: 129,064px² vs 24,191px² at t=0, exactly equal at t=0.5 (66,443px² each), reversed at t=1. At the midpoint the two co-heroes sit in separate rows (measured row tops 111.5px and 413.9px), each sharing its row with one small slot — slot 3 beside slot 2, slot 10 beside slot 11. The flow re-breaks rows continuously along the way; the test asserts validity at every step, the monotone swap, and the midpoint row relationship.

## Experiment 5 — text-aware fitness through pretext

pretext's `prepare()`/`layout()` runs inside the scoring loop: labels are prepared once, then each visible slot's label is line-broken at the slot's inner width and scored on whether the block height fits the slot. Bun has no canvas, so `prepare()` measures against a deterministic stub advance table installed as `OffscreenCanvas`; the line-breaking arithmetic is real pretext, only the advance numbers are synthetic (in a browser the same code measures for real). Measured: grid, heroRail, list, filmstrip all fit every label (1.000); the mosaic scores 0.938 because its smallest slot (importance 0.1, height preference 0.3, 42.9px tall) cannot hold even one 20px line after 12px insets — exactly the kind of judgement Duplo's text-density heuristics encoded. pretext (https://github.com/chenglou/pretext) isn't in the repo: `PRETEXT_SRC` names a folder holding its `src/` files (`lab/ui-exploration/text-tabulated/pretext/` is one), and the tests skip when the variable is unset, since loading pretext installs a stub `OffscreenCanvas` for the whole test process, which the text-tabulated and hn-thread tests, with fake canvases of their own, share under `bun test lab`.

## How every claim is verified

| claim | test |
| --- | --- |
| compiled tensor equals the reference interpreter on every live wire | `flow-layout.test.ts` — 5 families + 250 random vectors through `validateTensorEvaluation` |
| circuit matches the float64 oracle (geometry, ranks, bits, counts, depths) | `flow-layout.test.ts` — 400 random vectors, knife-edge margins skipped and bounded (<10%) |
| visible ranks always form a permutation; overlap never fires | same test, asserted per vector |
| tie rule, the inclusive 2^-36 band boundary, the adversarial collision | `flow-layout.test.ts` — exact ties, sub-ulp keys, one-ulp keys, 2^-36 vs 2^-35 spacing, the 1.3e-11 band |
| every wire stays inside its declared range, including under rank collisions | `flow-layout.test.ts` — collision with maximal heights checked against every declared range |
| five families valid, structurally distinct, zero recompilation (via the circuit's own tensor program property) | `flow-layout.test.ts` |
| fitness gradients | `fitness.test.ts` — central finite differences, 300+ checks |
| straight-through backward pass | `ste-backprop.test.ts` — equality with the reference pass off-boolean, pass-through on-boolean, layout-circuit probes vs finite differences |
| search results (1,078 valid; categories 0/2,886/44; top 0.875; determinism) | `generation.test.ts` |
| gradient: all 12 starts valid and improved; best 0.922 / median 0.887 / worst 0.865 pinned; beats search; 6-slot collapse | `generation.test.ts` |
| reward-hacking collapse without the richness term (0.998, 2 slots) | `generation.test.ts` |
| interpolation counts 1,906 / 4,154 repaired / 0 failed / 6,060; categories 0/4,154/0 | `interpolation.test.ts` — full 60x101 study re-run |
| hero migration valid throughout, monotone area transfer, midpoint rows | `interpolation.test.ts` |
| pretext text fit values | `text-fitness.test.ts` |

## Failures and limits

- The FP32 order-comparison band. Distinct keys whose difference is at most 2^-36 (~1.5e-11, inclusive: at exactly 2^-36 the score rounds back to 0.5 under round-to-even) make the pairwise comparison intransitive. The band is reachable only for keys up to 2^-12 (~2.44e-4), because FP32 spacing in [2^-13, 2^-12) is exactly 2^-36 and grows past the band above that. An adversarial triple inside the band produces colliding ranks and stacked slots. The circuit catches its own degeneracy — colliding slots trip the overlap check and the layout reports invalid — and the defensive span and height clamps keep every wire inside its declared range (the unclamped height sum used to escape its declared bound here; found and fixed during verification), but the layout itself is unusable. Exact ties (equal FP32 keys) are safe at any magnitude: they fall back to slot index deterministically. Characterized by test, including the exact 2^-36 boundary; never observed in any random or interpolated sample.
- Straight-through gradient bias. Exact on continuous paths, up to ~1.5x phantom gradient where fields feed frozen decisions (measured on the hero importance probe). Good enough for projection (2 steps) and for ascent direction; do not read the magnitudes as derivatives.
- Fitness under-specification. Gradient ascent finds 6-slot slab layouts scoring 0.922 that no designer would pick; the richness term bounded the collapse but did not eliminate it. The honest conclusion: search-with-prior and gradient optimize the same number and disagree about taste, so the fitness function is the bottleneck, which matches Flipboard needing dozens of terms plus Perlin-noise pacing.
- One row per slot. The flow cannot express a hero spanning multiple rows beside a stacked rail (the magazine hero-left layout); a hero owns exactly one row. That family needs either a second flow pass or a region split, both of which reintroduce structure that fields would have to select.
- No scrolling: the 16-slot list overflows the 800px viewport, so the authored list hides 10 slots. Overflow handling (scroll offset as a field) was out of scope.
- The moved-distance metric for projection is an aggregate; a max of 1.423 means at least one repaired point drifted noticeably from the straight line (still valid, still anchored by the proximal term). Per-point paths are available from `walkPath` if the drift distribution matters later.
- The pretext term uses a synthetic advance table under Bun (documented above) and a machine-local import path.

## Next steps

- Feed the projection into an actual transition: sample the repaired path at 60Hz and drive the DOM projection from the window-manager pattern, springs on top.
- Second flow region (hero column + rail flow) selected by fields, to reach the magazine family without forking code.
- Learn fitness weights instead of hand-tuning: the seven term values are already computed per layout, so fitting weights to pairwise human preferences is a 7-parameter logistic regression.
- Fold the scorer's remaining JS terms into circuit wires once dynamic division lands (fill and balance need one reciprocal each).
- Try bfloat16 for the whole circuit: every decision here is either an integer-exact clamp bit or a slack-guarded boolean, which is the recipe expressions.md says survives narrower formats.
