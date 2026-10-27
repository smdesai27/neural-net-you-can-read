// Fork of vendor/handwired/lab/generative-fields/flow-layout.ts. Two changes:
// - A tile's span is at least minTilePx wide: span = min(1, max(raw, minTilePx *
//   invInnerWidth)), so the column count changes with the viewport width.
// - The builder tags wires with what they compute (see tag() and labels.ts), so
//   the lens can name every neuron.
// Everything else is upstream's circuit, comments included.
//
// One equivariant flow layout circuit over 16 slots.
//
// Layout model: each slot turns its fields into a width span (fraction of the
// inner width) and a pixel height through shared coefficients. Slots are
// ordered by their order key (rank computed in-circuit from pairwise
// comparisons, ties broken by slot index), then placed by greedy line
// breaking: a slot that would overflow the current row starts a new row, and a
// row is as tall as its tallest slot. The same sub-circuit runs for every
// slot; the only thing distinguishing slots is their field values and the slot
// index entering as the tie-break.
//
// The circuit also scores its own output: exact violation counts (pairwise
// overlap, viewport containment, minimum sizes) with an aggregate valid bit,
// plus a continuous violation depth in pixels that projection can descend.
//
// Decision style follows expressions.md: counts use integer-exact clamp01
// decision bits (rank equality, violation aggregation); genuinely continuous
// comparisons (order keys, wrap tests, pixel thresholds) use the boolean atom.

import {
  Circuit,
  type Wire,
  emitMaximum,
  emitMinimum,
  emitMultiply,
  emitRelu,
  emitSelect,
  invariant,
  validateCircuit,
  validateTensorProgram,
  validateWireView,
} from '../../vendor/handwired/src/runtime/numeric-ui-core.ts'
import {
  extendedSourceRanges,
  extendedSourceSize,
  globalField,
  invInnerWidthCell,
  minTilePxCell,
  slotCount,
  slotField,
  slotFieldIndex,
} from './schema.ts'
import {type Stage, type WireTags, createTagger} from './labels.ts'

// Pixel comparisons are scaled by this before the boolean rounding step; it
// keeps scores near 0.5 for deficits up to tens of thousands of pixels while
// resolving differences well below a pixel.
const pixelScoreScale = 1 / 65536

// A violation must exceed this many pixels before its exact bit fires. The
// continuous violation depth still reports smaller overshoots.
export const violationSlackPixels = 0.5

export const geometryField = {
  x: 0,
  y: 1,
  width: 2,
  height: 3,
  size: 4,
} as const

export type FlowLayoutOutput = {
  // Per-slot pixel geometry, all zero for hidden slots: slotCount * geometryField.size wires.
  slotGeometry: Wire[]
  // Exact 0/1 visibility per slot.
  visibleBits: Wire[]
  // Exact integer rank per slot among visible slots (unused for hidden slots).
  ranks: Wire[]
  overlapViolationCount: Wire
  containViolationCount: Wire
  sizeViolationCount: Wire
  violationCount: Wire
  validBit: Wire
  // Continuous pixel measure of all violations; zero exactly when nothing
  // overshoots at all, small when only sub-slack overshoots remain.
  violationDepth: Wire
  coveredArea: Wire
  contentBottom: Wire
  // The score each rank position's wrap decision rounds: at least 0.5 wraps.
  wrapScores: Wire[]
  wrapBits: Wire[]
}

// clamp01 with a constant unit, so it needs no unit wire.
function emitClamp01(circuit: Circuit, value: Wire): Wire {
  const positive = emitRelu(circuit, value)
  const aboveOne = emitRelu(circuit, circuit.affine([[value, 1]], -1))
  const result = circuit.affine([[positive, 1], [aboveOne, -1]])
  circuit.assumeRange(result, 0, 1)
  return result
}

// Exact bit for `count >= threshold` when count only takes integer values
// (expressions.md "Decision bits"): an integer cannot land strictly inside
// the clamp ramp, so no boolean atom is needed.
function emitCountAtLeastOne(circuit: Circuit, count: Wire): Wire {
  return emitClamp01(circuit, count)
}

// Bit for `value >= threshold` on a genuinely continuous value. The score sits
// at 0.5 exactly on the threshold, so the boolean atom rounds equality up.
function emitThresholdBit(circuit: Circuit, value: Wire, threshold: number, scale: number): Wire {
  const score = emitClamp01(
    circuit,
    circuit.affine([[value, scale]], 0.5 - threshold * scale),
  )
  return circuit.boolean(score)
}

// Bit for `keyLater >= keyEarlier`, built so that near-equal keys stay
// well-behaved. The two relu wires reconstruct the exact FP32 difference
// (Sterbenz: subtraction of nearly equal floats is exact), and the 1024x scale
// shrinks the rounding band around 0.5 to key differences of at most 2^-36
// (~1.5e-11); the boundary is inclusive, because 0.5 - 2^-26 rounds back up to
// 0.5 under round-to-even. Key differences inside the band behave as ties.
// Exactly equal keys give score 0.5, which rounds to 1: the earlier slot wins.
function emitKeyOrderBit(circuit: Circuit, keyEarlier: Wire, keyLater: Wire): Wire {
  const laterAbove = emitRelu(circuit, circuit.affine([[keyLater, 1], [keyEarlier, -1]]))
  const earlierAbove = emitRelu(circuit, circuit.affine([[keyEarlier, 1], [keyLater, -1]]))
  const score = emitClamp01(
    circuit,
    circuit.affine([[laterAbove, 1024], [earlierAbove, -1024]], 0.5),
  )
  return circuit.boolean(score)
}

export function createFlowLayoutCircuit(): {circuit: Circuit, output: FlowLayoutOutput, tags: WireTags} {
  const circuit = new Circuit(extendedSourceSize)
  for (let wire = 0; wire < extendedSourceSize; wire++) {
    const [minimum, maximum] = extendedSourceRanges[wire]!
    circuit.assumeRange(wire, minimum, maximum)
  }
  const {tags, tag} = createTagger()
  const tagEach = (wires: readonly Wire[], stage: Stage, label: (index: number) => string, slot?: (index: number) => number) =>
    wires.forEach((wire, index) => tag(wire, stage, label(index), slot?.(index)))

  const zero = circuit.affine([])
  const importance = Array.from({length: slotCount}, (_, slot) => slotFieldIndex(slot, slotField.importance))
  const orderKey = Array.from({length: slotCount}, (_, slot) => slotFieldIndex(slot, slotField.orderKey))
  const visibility = Array.from({length: slotCount}, (_, slot) => slotFieldIndex(slot, slotField.visibility))
  const heightPreference = Array.from({length: slotCount}, (_, slot) => slotFieldIndex(slot, slotField.heightPreference))

  // Visibility hardens at 0.5. Continuous field, so the boolean atom applies.
  const visibleBits = visibility.map(wire => circuit.boolean(wire))
  tagEach(visibleBits, 'order', slot => `tile ${slot + 1}: shown?`, slot => slot)

  // Ranks. One comparison bit per unordered pair; the other direction is its
  // complement, so each pair contributes to exactly one slot's rank and the
  // visible ranks form a permutation of 0..visibleCount-1.
  const pairEarlierFirst: Wire[][] = Array.from({length: slotCount}, () => [])
  for (let earlier = 0; earlier < slotCount; earlier++) {
    for (let later = earlier + 1; later < slotCount; later++) {
      pairEarlierFirst[earlier]![later] = emitKeyOrderBit(
        circuit,
        orderKey[earlier]!,
        orderKey[later]!,
      )
      tag(pairEarlierFirst[earlier]![later]!, 'order', `tile ${earlier + 1} comes before tile ${later + 1}?`)
    }
  }
  const ranks = Array.from({length: slotCount}, (_, slot) => {
    const terms: [Wire, number][] = []
    for (let other = 0; other < slotCount; other++) {
      if (other === slot) continue
      // A visible `other` counts toward this slot's rank when it comes first.
      const otherFirst = other < slot
        ? pairEarlierFirst[other]![slot]!
        : circuit.affine([[pairEarlierFirst[slot]![other]!, -1]], 1)
      terms.push([emitMultiply(circuit, visibleBits[other]!, otherFirst), 1])
    }
    const rank = circuit.affine(terms)
    circuit.assumeRange(rank, 0, slotCount - 1)
    tag(rank, 'order', `tile ${slot + 1}: place in reading order`, slot)
    return rank
  })

  // One-hot rank membership: rankIs[slot][position] is the exact integer bit
  // `rank == position`, built from clamp01 ramps over the integer rank.
  const rankIs: Wire[][] = Array.from({length: slotCount}, (_, slot) => {
    const rankAtLeast: Wire[] = [circuit.affine([], 1)]
    for (let position = 1; position < slotCount; position++) {
      rankAtLeast.push(emitClamp01(
        circuit,
        circuit.affine([[ranks[slot]!, 1]], 1 - position),
      ))
    }
    return Array.from({length: slotCount}, (_, position) => {
      if (position === slotCount - 1) return rankAtLeast[position]!
      const bit = circuit.affine([
        [rankAtLeast[position]!, 1],
        [rankAtLeast[position + 1]!, -1],
      ])
      circuit.assumeRange(bit, 0, 1)
      tag(bit, 'order', `tile ${slot + 1} is in place ${position + 1}?`, slot)
      return bit
    })
  })

  // rankSelect[slot][position] gates a visible slot into its rank position.
  const rankSelect: Wire[][] = Array.from({length: slotCount}, (_, slot) =>
    Array.from({length: slotCount}, (_, position) =>
      emitMultiply(circuit, visibleBits[slot]!, rankIs[slot]![position]!)))

  // Per-slot span (width as a fraction of the inner width) and pixel height,
  // both from importance through shared global coefficients. The fork's floor:
  // a span is at least minTilePx as a fraction of the inner width.
  const minSpan = emitMultiply(circuit, minTilePxCell, invInnerWidthCell)
  tag(minSpan, 'size', 'narrowest tile, as a fraction of the row')
  const spans = Array.from({length: slotCount}, (_, slot) => {
    const preferred = circuit.affine([
      [globalField.baseSpan, 1],
      [emitMultiply(circuit, globalField.spanFromImportance, importance[slot]!), 1],
    ])
    tag(preferred, 'size', `tile ${slot + 1}: preferred width`, slot)
    const raw = emitMaximum(circuit, preferred, minSpan)
    tag(raw, 'size', `tile ${slot + 1}: width after the minimum`, slot)
    const span = circuit.affine([
      [raw, 1],
      [emitRelu(circuit, circuit.affine([[raw, 1]], -1)), -1],
    ])
    circuit.assumeRange(span, 0.02, 1)
    tag(span, 'size', `tile ${slot + 1}: width (fraction of the row)`, slot)
    return span
  })
  const heights = Array.from({length: slotCount}, (_, slot) => {
    const base = circuit.affine([
      [globalField.baseHeight, 1],
      [emitMultiply(circuit, globalField.heightFromImportance, importance[slot]!), 1],
    ])
    const multiplier = circuit.affine([[heightPreference[slot]!, 1]], 0.5)
    const height = emitMultiply(circuit, base, multiplier)
    tag(height, 'size', `tile ${slot + 1}: height in px`, slot)
    return height
  })

  // Fields of the slot occupying each rank position. The rankSelect bits are
  // one-hot across slots for a fixed position (visible ranks are a
  // permutation), so each sum passes one slot's value through exactly; the
  // FP32 additions of zeros lose nothing. The width and height clamps below
  // are defensive: if order keys ever land inside the comparison rounding band
  // and two slots collide on one rank, the raw sums would carry two slots'
  // spans and heights past any single slot's bounds. The clamps keep every
  // downstream wire inside its declared range (the clamp is an exact identity
  // for legitimate one-hot sums) while the overlap check reports the collision.
  const spanAtPosition: Wire[] = []
  const heightAtPosition: Wire[] = []
  // The tallest any single slot can be under the declared source ranges:
  // (baseHeight 1024 + heightFromImportance 1024) * multiplier 1.5.
  const slotHeightLimit = 3072
  for (let position = 0; position < slotCount; position++) {
    const spanSum = circuit.affine(
      Array.from({length: slotCount}, (_, slot): [Wire, number] =>
        [emitMultiply(circuit, rankSelect[slot]![position]!, spans[slot]!), 1]),
    )
    const clampedSpan = circuit.affine([
      [spanSum, 1],
      [emitRelu(circuit, circuit.affine([[spanSum, 1]], -1)), -1],
    ])
    circuit.assumeRange(clampedSpan, 0, 1)
    tag(clampedSpan, 'flow', `place ${position + 1}: width of its tile`)
    spanAtPosition.push(clampedSpan)
    const heightSum = circuit.affine(
      Array.from({length: slotCount}, (_, slot): [Wire, number] =>
        [emitMultiply(circuit, rankSelect[slot]![position]!, heights[slot]!), 1]),
    )
    const clampedHeight = circuit.affine([
      [heightSum, 1],
      [emitRelu(circuit, circuit.affine([[heightSum, 1]], -slotHeightLimit)), -1],
    ])
    circuit.assumeRange(clampedHeight, 0, slotHeightLimit)
    tag(clampedHeight, 'flow', `place ${position + 1}: height of its tile`)
    heightAtPosition.push(clampedHeight)
  }

  // Greedy line breaking over rank positions. All horizontal quantities are
  // fractions of the inner width; vertical quantities are pixels. Empty rank
  // positions carry zero span and height, so they never move the row downward
  // in a way any visible slot reads.
  const xAtPosition: Wire[] = []
  const rowTopAtPosition: Wire[] = []
  const wrapScores: Wire[] = []
  const wrapBits: Wire[] = []
  let cursor = zero
  let rowTop: Wire = globalField.paddingY
  let rowBottom: Wire = globalField.paddingY
  for (let position = 0; position < slotCount; position++) {
    const span = spanAtPosition[position]!
    const height = heightAtPosition[position]!
    // Wrap when cursor + span >= 1 + wrapSlack; score 0.5 exactly on the
    // threshold rounds to a wrap.
    const wrapScore = emitClamp01(circuit, circuit.affine([
      [cursor, 0.5],
      [span, 0.5],
      [globalField.wrapSlack, -0.5],
    ]))
    const wrap = circuit.boolean(wrapScore)
    tag(wrapScore, 'flow', `place ${position + 1}: how close to wrapping (0.5 = the edge)`)
    tag(wrap, 'flow', `place ${position + 1}: starts a new row?`)
    wrapScores.push(wrapScore)
    wrapBits.push(wrap)
    const x = emitSelect(circuit, wrap, cursor, zero)
    circuit.assumeRange(x, 0, 2)
    tag(x, 'flow', `place ${position + 1}: left edge (fraction of the row)`)
    const top = emitSelect(
      circuit,
      wrap,
      rowTop,
      circuit.affine([[rowBottom, 1], [globalField.gapY, 1]]),
    )
    circuit.assumeRange(top, 0, 65536)
    tag(top, 'flow', `place ${position + 1}: top of its row`)
    const bottomCandidate = circuit.affine([[top, 1], [height, 1]])
    const bottom = emitSelect(
      circuit,
      wrap,
      emitMaximum(circuit, rowBottom, bottomCandidate),
      bottomCandidate,
    )
    circuit.assumeRange(bottom, 0, 65536)
    tag(bottom, 'flow', `place ${position + 1}: bottom of the row so far`)
    xAtPosition.push(x)
    rowTopAtPosition.push(top)
    cursor = circuit.affine([[x, 1], [span, 1], [globalField.gapX, 1]])
    circuit.assumeRange(cursor, 0, 3)
    tag(cursor, 'flow', `place ${position + 1}: where the next tile would start`)
    rowTop = top
    rowBottom = bottom
  }

  // Route flow results back to slots through the same one-hot bits, then
  // convert fractions to pixels and gate everything by visibility.
  const innerWidth = emitRelu(circuit, circuit.affine([
    [globalField.viewportWidth, 1],
    [globalField.paddingX, -2],
  ]))
  const slotGeometry: Wire[] = []
  const rightEdges: Wire[] = []
  const bottomEdges: Wire[] = []
  const widthPixelsUngated: Wire[] = []
  const heightPixelsUngated: Wire[] = []
  for (let slot = 0; slot < slotCount; slot++) {
    const xFraction = circuit.affine(
      Array.from({length: slotCount}, (_, position): [Wire, number] =>
        [emitMultiply(circuit, rankSelect[slot]![position]!, xAtPosition[position]!), 1]),
    )
    circuit.assumeRange(xFraction, 0, 2)
    const yPixels = circuit.affine(
      Array.from({length: slotCount}, (_, position): [Wire, number] =>
        [emitMultiply(circuit, rankSelect[slot]![position]!, rowTopAtPosition[position]!), 1]),
    )
    circuit.assumeRange(yPixels, 0, 65536)
    const xPixels = circuit.affine([
      [globalField.paddingX, 1],
      [emitMultiply(circuit, innerWidth, xFraction), 1],
    ])
    const widthPixels = emitMultiply(circuit, innerWidth, spans[slot]!)
    const x = emitMultiply(circuit, visibleBits[slot]!, xPixels)
    const y = emitMultiply(circuit, visibleBits[slot]!, yPixels)
    const width = emitMultiply(circuit, visibleBits[slot]!, widthPixels)
    const height = emitMultiply(circuit, visibleBits[slot]!, heights[slot]!)
    slotGeometry.push(x, y, width, height)
    tag(x, 'placement', `tile ${slot + 1}: final x`, slot)
    tag(y, 'placement', `tile ${slot + 1}: final y`, slot)
    tag(width, 'placement', `tile ${slot + 1}: final width`, slot)
    tag(height, 'placement', `tile ${slot + 1}: final height`, slot)
    rightEdges.push(circuit.affine([[x, 1], [width, 1]]))
    bottomEdges.push(circuit.affine([[y, 1], [height, 1]]))
    widthPixelsUngated.push(widthPixels)
    heightPixelsUngated.push(heights[slot]!)
  }

  // Pairwise overlap among visible slots. By construction the flow cannot
  // overlap (the cursor only advances, rows clear the tallest member), so
  // these wires double as a permanent self-check; the tests keep them honest.
  const overlapBits: Wire[] = []
  const overlapDepths: Wire[] = []
  for (let earlier = 0; earlier < slotCount; earlier++) {
    for (let later = earlier + 1; later < slotCount; later++) {
      const penetrationX = circuit.affine([
        [emitMinimum(circuit, rightEdges[earlier]!, rightEdges[later]!), 1],
        [emitMaximum(circuit, slotGeometry[earlier * geometryField.size + geometryField.x]!, slotGeometry[later * geometryField.size + geometryField.x]!), -1],
      ])
      const penetrationY = circuit.affine([
        [emitMinimum(circuit, bottomEdges[earlier]!, bottomEdges[later]!), 1],
        [emitMaximum(circuit, slotGeometry[earlier * geometryField.size + geometryField.y]!, slotGeometry[later * geometryField.size + geometryField.y]!), -1],
      ])
      const penetration = emitRelu(
        circuit,
        emitMinimum(circuit, penetrationX, penetrationY),
      )
      const bothVisible = emitMultiply(circuit, visibleBits[earlier]!, visibleBits[later]!)
      const depth = emitMultiply(circuit, bothVisible, penetration)
      overlapDepths.push(depth)
      overlapBits.push(emitThresholdBit(circuit, depth, violationSlackPixels, pixelScoreScale))
      tag(overlapBits.at(-1)!, 'checks', `tiles ${earlier + 1} and ${later + 1} overlap?`)
    }
  }
  const overlapViolationCount = circuit.affine(overlapBits.map((bit): [Wire, number] => [bit, 1]))
  circuit.assumeRange(overlapViolationCount, 0, overlapBits.length)

  // Containment. Left and top edges cannot underflow (they start at the
  // padding), so only the bottom and right edges are checked. The right edge
  // gets the wrap slack as allowance: a row is allowed to run that far by the
  // wrap rule, so only overshoot beyond it counts as a violation.
  const bottomLimit = circuit.affine([
    [globalField.viewportHeight, 1],
    [globalField.paddingY, -1],
  ])
  const rightLimit = circuit.affine([
    [globalField.paddingX, 1],
    [innerWidth, 1],
    [emitMultiply(circuit, innerWidth, globalField.wrapSlack), 1],
  ])
  const containBits: Wire[] = []
  const containDepths: Wire[] = []
  for (let slot = 0; slot < slotCount; slot++) {
    const bottomOvershoot = emitMultiply(
      circuit,
      visibleBits[slot]!,
      emitRelu(circuit, circuit.affine([[bottomEdges[slot]!, 1], [bottomLimit, -1]])),
    )
    const rightOvershoot = emitMultiply(
      circuit,
      visibleBits[slot]!,
      emitRelu(circuit, circuit.affine([[rightEdges[slot]!, 1], [rightLimit, -1]])),
    )
    containDepths.push(bottomOvershoot, rightOvershoot)
    containBits.push(
      emitThresholdBit(circuit, bottomOvershoot, violationSlackPixels, pixelScoreScale),
      emitThresholdBit(circuit, rightOvershoot, violationSlackPixels, pixelScoreScale),
    )
    tag(containBits.at(-2)!, 'checks', `tile ${slot + 1}: runs off the bottom?`, slot)
    tag(containBits.at(-1)!, 'checks', `tile ${slot + 1}: runs off the right?`, slot)
  }
  const containViolationCount = circuit.affine(containBits.map((bit): [Wire, number] => [bit, 1]))
  circuit.assumeRange(containViolationCount, 0, containBits.length)

  // Minimum sizes for visible slots, measured on the ungated pixel sizes so a
  // hidden slot (geometry zero) does not count.
  const sizeBits: Wire[] = []
  const sizeDepths: Wire[] = []
  for (let slot = 0; slot < slotCount; slot++) {
    const widthDeficit = emitMultiply(
      circuit,
      visibleBits[slot]!,
      emitRelu(circuit, circuit.affine([
        [globalField.minVisibleWidth, 1],
        [widthPixelsUngated[slot]!, -1],
      ])),
    )
    const heightDeficit = emitMultiply(
      circuit,
      visibleBits[slot]!,
      emitRelu(circuit, circuit.affine([
        [globalField.minVisibleHeight, 1],
        [heightPixelsUngated[slot]!, -1],
      ])),
    )
    sizeDepths.push(widthDeficit, heightDeficit)
    sizeBits.push(
      emitThresholdBit(circuit, widthDeficit, violationSlackPixels, pixelScoreScale),
      emitThresholdBit(circuit, heightDeficit, violationSlackPixels, pixelScoreScale),
    )
    tag(sizeBits.at(-2)!, 'checks', `tile ${slot + 1}: too narrow?`, slot)
    tag(sizeBits.at(-1)!, 'checks', `tile ${slot + 1}: too short?`, slot)
  }
  const sizeViolationCount = circuit.affine(sizeBits.map((bit): [Wire, number] => [bit, 1]))
  circuit.assumeRange(sizeViolationCount, 0, sizeBits.length)

  const violationCount = circuit.affine([
    [overlapViolationCount, 1],
    [containViolationCount, 1],
    [sizeViolationCount, 1],
  ])
  circuit.assumeRange(violationCount, 0, overlapBits.length + containBits.length + sizeBits.length)
  const validBit = circuit.affine([[emitCountAtLeastOne(circuit, violationCount), -1]], 1)
  circuit.assumeRange(validBit, 0, 1)
  tag(violationCount, 'checks', 'number of broken rules')
  tag(validBit, 'checks', 'layout is valid?')

  const violationDepth = circuit.affine([
    ...overlapDepths.map((depth): [Wire, number] => [depth, 1]),
    ...containDepths.map((depth): [Wire, number] => [depth, 1]),
    ...sizeDepths.map((depth): [Wire, number] => [depth, 1]),
  ])

  // Fitness numerators the circuit can compute without division: total covered
  // area and the lowest visible bottom edge. The scorer in fitness.ts owns
  // every division (fill ratios, means, correlations), following the repo rule
  // that the parameter/scorer side owns division.
  const coveredArea = circuit.affine(
    Array.from({length: slotCount}, (_, slot): [Wire, number] => [
      emitMultiply(
        circuit,
        slotGeometry[slot * geometryField.size + geometryField.width]!,
        slotGeometry[slot * geometryField.size + geometryField.height]!,
      ),
      1,
    ]),
  )
  let contentBottom = bottomEdges[0]!
  for (let slot = 1; slot < slotCount; slot++) {
    contentBottom = emitMaximum(circuit, contentBottom, bottomEdges[slot]!)
  }

  tag(violationDepth, 'checks', 'total px of rule-breaking')
  tag(coveredArea, 'checks', 'area covered by tiles')
  tag(contentBottom, 'checks', 'bottom of the lowest tile')
  return {
    circuit,
    tags,
    output: {
      wrapScores,
      wrapBits,
      slotGeometry,
      visibleBits,
      ranks,
      overlapViolationCount,
      containViolationCount,
      sizeViolationCount,
      violationCount,
      validBit,
      violationDepth,
      coveredArea,
      contentBottom,
    },
  }
}

export function flowLayoutOutputWires(output: FlowLayoutOutput): Wire[] {
  return [
    ...output.slotGeometry,
    ...output.visibleBits,
    ...output.ranks,
    output.overlapViolationCount,
    output.containViolationCount,
    output.sizeViolationCount,
    output.violationCount,
    output.validBit,
    output.violationDepth,
    output.coveredArea,
    output.contentBottom,
    ...output.wrapScores,
    ...output.wrapBits,
  ]
}

export function createFlowLayoutMachine() {
  const {circuit, output, tags} = createFlowLayoutCircuit()
  const wireCount = validateCircuit(circuit)
  validateWireView(output.slotGeometry, slotCount * geometryField.size, wireCount)
  validateWireView(output.visibleBits, slotCount, wireCount)
  validateWireView(output.ranks, slotCount, wireCount)
  const tensorProgram = circuit.tensorProgram = circuit.compileTensorProgram(
    flowLayoutOutputWires(output),
  )
  validateTensorProgram(circuit, tensorProgram)
  return {
    circuit,
    output,
    tags,
    tensorProgram,
    evaluate(source: Float32Array): Float32Array {
      return circuit.evaluateTensor(source)
    },
    evaluateReference(source: Float32Array): Float32Array {
      return circuit.evaluateReference(source)
    },
  }
}

export type FlowLayoutMachine = ReturnType<typeof createFlowLayoutMachine>

// Convenience readers over an evaluated values array.
export function readSlotGeometry(
  values: Float32Array,
  output: FlowLayoutOutput,
  slot: number,
): {x: number, y: number, width: number, height: number} {
  invariant(Number.isInteger(slot) && slot >= 0 && slot < slotCount, 'slot index is invalid')
  const start = slot * geometryField.size
  return {
    x: values[output.slotGeometry[start + geometryField.x]!]!,
    y: values[output.slotGeometry[start + geometryField.y]!]!,
    width: values[output.slotGeometry[start + geometryField.width]!]!,
    height: values[output.slotGeometry[start + geometryField.height]!]!,
  }
}
