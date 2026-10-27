import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {validateTensorEvaluation} from '../../src/runtime/numeric-ui-core.ts'
import {
  authoredFamilies,
  buildSource,
  defaultContext,
  fieldDimension,
  slotCount,
} from './fields.ts'
import {createFlowLayoutMachine, readSlotGeometry} from './flow-layout.ts'
import {evaluateOracle} from './oracle.ts'
import {createRandom, sampleFieldVector} from './search.ts'

const machine = createFlowLayoutMachine()

function evaluateFields(fields: Float64Array) {
  const source = buildSource(fields, defaultContext)
  return {source, values: machine.evaluate(source)}
}

test('authored families are valid and structurally distinct on one compiled circuit', () => {
  const operationCount = machine.circuit.operations.length
  const tensorProgram = machine.tensorProgram
  const families = new Map(
    authoredFamilies().map(({name, fields}) => {
      const {values} = evaluateFields(fields)
      return [name, values]
    }),
  )

  for (const [name, values] of families) {
    assert.equal(values[machine.output.validBit], 1, `${name} must be valid`)
    assert.equal(values[machine.output.violationCount], 0, `${name} must have zero violations`)
  }

  // Zero recompilation: evaluating all five families added no operations and
  // kept the same compiled tensor program object. The check must read the
  // circuit's own tensorProgram property: appending an operation or declaring
  // a one-hot group nulls that property and evaluateTensor then recompiles
  // into a fresh object, while machine.tensorProgram is a snapshot that never
  // changes and would compare equal to itself vacuously.
  assert.equal(machine.circuit.operations.length, operationCount)
  assert.equal(machine.circuit.tensorProgram, tensorProgram)

  function geometry(name: string, slot: number) {
    return readSlotGeometry(families.get(name)!, machine.output, slot)
  }
  function visibleSlots(name: string) {
    const values = families.get(name)!
    return Array.from({length: slotCount}, (_, slot) => slot)
      .filter(slot => values[machine.output.visibleBits[slot]!] === 1)
  }
  function distinctRowTops(name: string) {
    const tops = new Set<number>()
    for (const slot of visibleSlots(name)) tops.add(Math.round(geometry(name, slot).y))
    return tops
  }

  // Grid: 16 visible in 4 rows of 4, equal widths, ranks in slot order.
  assert.equal(visibleSlots('grid').length, 16)
  assert.equal(distinctRowTops('grid').size, 4)
  const gridWidth = geometry('grid', 0).width
  for (let slot = 0; slot < slotCount; slot++) {
    assert.ok(Math.abs(geometry('grid', slot).width - gridWidth) < 0.01)
    assert.equal(families.get('grid')![machine.output.ranks[slot]!], slot)
  }

  // Hero rail: the importance peak owns its row and dwarfs the rail slots.
  const heroGeometry = geometry('heroRail', 0)
  const heroArea = heroGeometry.width * heroGeometry.height
  for (let slot = 1; slot < slotCount; slot++) {
    const railGeometry = geometry('heroRail', slot)
    assert.ok(heroArea > 2.5 * railGeometry.width * railGeometry.height)
    assert.ok(Math.abs(railGeometry.y - heroGeometry.y) > 1, 'hero must own its row')
  }

  // Mosaic: varied widths across several rows.
  const mosaicWidths = new Set(
    visibleSlots('mosaic').map(slot => Math.round(geometry('mosaic', slot).width)),
  )
  assert.ok(mosaicWidths.size >= 4)
  assert.ok(distinctRowTops('mosaic').size >= 5)

  // List: six full-width slots, one per row; the rest hidden with zero geometry.
  assert.deepEqual(visibleSlots('list'), [0, 1, 2, 3, 4, 5])
  assert.equal(distinctRowTops('list').size, 6)
  for (const slot of visibleSlots('list')) {
    assert.ok(Math.abs(geometry('list', slot).width - 1200) < 0.5)
  }
  for (let slot = 6; slot < slotCount; slot++) {
    assert.deepEqual(geometry('list', slot), {x: 0, y: 0, width: 0, height: 0})
  }

  // Filmstrip: all 16 in one row.
  assert.equal(visibleSlots('filmstrip').length, 16)
  assert.equal(distinctRowTops('filmstrip').size, 1)
})

test('compiled tensor evaluation matches the reference interpreter on every live wire', () => {
  for (const {fields} of authoredFamilies()) {
    const {source, values} = evaluateFields(fields)
    validateTensorEvaluation(source, machine.circuit, values)
  }
  const random = createRandom(0x7E57_0001)
  for (let trial = 0; trial < 250; trial++) {
    const {source, values} = evaluateFields(sampleFieldVector(random))
    validateTensorEvaluation(source, machine.circuit, values)
  }
})

test('circuit agrees with the float64 oracle across seeded random field vectors', () => {
  const seededRandom = createRandom(0x02AC_1E00)
  const trials = 400
  let knifeEdgeSkips = 0
  for (let trial = 0; trial < trials; trial++) {
    const fields = sampleFieldVector(seededRandom)
    const {source, values} = evaluateFields(fields)
    const oracle = evaluateOracle(source)

    // Skip decision assertions when the oracle saw a knife-edge margin the
    // FP32 circuit may resolve differently; geometry stays asserted.
    const clearMargins =
      oracle.margins.orderKey > 1e-6
      && oracle.margins.visibility > 1e-4
      && oracle.margins.wrap > 1e-5
      && oracle.margins.violation > 0.02
    if (!clearMargins) {
      knifeEdgeSkips++
      continue
    }

    const visibleRanks: number[] = []
    for (let slot = 0; slot < slotCount; slot++) {
      const oracleSlot = oracle.slots[slot]!
      assert.equal(
        values[machine.output.visibleBits[slot]!],
        oracleSlot.visible ? 1 : 0,
      )
      assert.equal(values[machine.output.ranks[slot]!], oracleSlot.rank)
      if (oracleSlot.visible) visibleRanks.push(values[machine.output.ranks[slot]!]!)
      const circuitGeometry = readSlotGeometry(values, machine.output, slot)
      assert.ok(Math.abs(circuitGeometry.x - oracleSlot.x) < 0.02)
      assert.ok(Math.abs(circuitGeometry.y - oracleSlot.y) < 0.02)
      assert.ok(Math.abs(circuitGeometry.width - oracleSlot.width) < 0.02)
      assert.ok(Math.abs(circuitGeometry.height - oracleSlot.height) < 0.02)
    }

    // Visible ranks form a permutation of 0..visibleCount-1.
    assert.deepEqual(
      [...visibleRanks].sort((a, b) => a - b),
      Array.from({length: visibleRanks.length}, (_, rank) => rank),
    )

    assert.equal(values[machine.output.overlapViolationCount], oracle.overlapViolationCount)
    assert.equal(values[machine.output.containViolationCount], oracle.containViolationCount)
    assert.equal(values[machine.output.sizeViolationCount], oracle.sizeViolationCount)
    assert.equal(values[machine.output.validBit], oracle.valid ? 1 : 0)

    // The flow construction cannot overlap; the check must never fire.
    assert.equal(values[machine.output.overlapViolationCount], 0)

    const depthScale = Math.max(1, Math.abs(oracle.violationDepth))
    assert.ok(
      Math.abs(values[machine.output.violationDepth]! - oracle.violationDepth)
        < 0.05 + 1e-4 * depthScale,
    )
    assert.ok(
      Math.abs(values[machine.output.coveredArea]! - oracle.coveredArea)
        < 1 + 1e-4 * Math.abs(oracle.coveredArea),
    )
    assert.ok(Math.abs(values[machine.output.contentBottom]! - oracle.contentBottom) < 0.02)
  }
  assert.ok(
    knifeEdgeSkips < trials * 0.1,
    `too many knife-edge skips: ${knifeEdgeSkips}/${trials}`,
  )
})

test('order key ties fall back to slot index, and the FP32 comparison band is caught by validity', () => {
  const grid = authoredFamilies()[0]!.fields

  function ranksWith(keys: number[]) {
    const fields = new Float64Array(grid)
    for (let slot = 0; slot < slotCount; slot++) {
      fields[fieldDimension(`slot${slot}.orderKey`)] = keys[slot]!
    }
    const {values} = evaluateFields(fields)
    return {
      ranks: machine.output.ranks.map(wire => values[wire]!),
      valid: values[machine.output.validBit]!,
      overlap: values[machine.output.overlapViolationCount]!,
    }
  }

  const indexOrder = Array.from({length: slotCount}, (_, slot) => slot)

  // Exact ties: slot index orders the slots.
  assert.deepEqual(
    ranksWith(Array.from({length: slotCount}, () => 0.5)).ranks,
    indexOrder,
  )

  // Differences below FP32 resolution collapse to exact ties in the source
  // table, so they also fall back to index order.
  assert.deepEqual(
    ranksWith(indexOrder.map(slot => 0.5 + slot * 1e-9)).ranks,
    indexOrder,
  )

  // One FP32 ulp apart is already a strict, correctly resolved order.
  assert.deepEqual(
    ranksWith(indexOrder.map(slot => {
      let key = 0.5
      for (let step = 0; step < slot; step++) key = Math.fround(key + 2 ** -24)
      return key
    })).ranks,
    indexOrder,
  )

  // Distinct keys ~1e-9 apart (above the ~1.5e-11 comparison band) reverse
  // the order correctly even at tiny magnitudes.
  const reversed = [...indexOrder].reverse()
  assert.deepEqual(
    ranksWith(indexOrder.map(slot => (slotCount - 1 - slot) * 1e-9)).ranks,
    reversed,
  )

  // Inside the band (~1.3e-11 spacing, below 2^-36) the pairwise comparison
  // is no longer transitive and ranks can collide. The layout is degenerate,
  // but the circuit reports the degeneracy itself: colliding slots stack and
  // the overlap check fires.
  const band = ranksWith(indexOrder.map(slot => (slotCount - 1 - slot) * 1.3e-11))
  const rankSet = new Set(band.ranks)
  assert.ok(rankSet.size < slotCount, 'the band case is expected to collide')
  assert.equal(band.valid, 0)
  assert.ok(band.overlap > 0)

  // The band boundary is 2^-36 inclusive: at exactly that difference the
  // score 0.5 - 2^-26 rounds back up to 0.5 under round-to-even, so the pair
  // ties and slot index decides. One binade further (2^-35) resolves strictly.
  const farKeys = indexOrder.map(slot => (slot + 0.5) / slotCount)
  const atBoundary = [...farKeys]
  atBoundary[0] = 2 ** -36
  atBoundary[1] = 0
  const boundaryRanks = ranksWith(atBoundary).ranks
  assert.ok(
    boundaryRanks[0]! < boundaryRanks[1]!,
    'a difference of exactly 2^-36 must behave as a tie (index order)',
  )
  const pastBoundary = [...farKeys]
  pastBoundary[0] = 2 ** -35
  pastBoundary[1] = 0
  const pastRanks = ranksWith(pastBoundary).ranks
  assert.ok(
    pastRanks[1]! < pastRanks[0]!,
    'a difference of 2^-35 must resolve strictly by key',
  )
})

// The rank-collision degeneracy must not leave any wire outside its declared
// range: two slots on one rank position would push the raw span and height
// sums past any single slot's bounds if the defensive clamps were missing.
// This drives the collision with the tallest slots the trainer-facing field
// ranges allow, then checks every wire of the reference evaluation against the
// range the builder declared for it.
test('every wire stays inside its declared range, including under rank collisions', () => {
  function assertWiresWithinDeclaredRanges(values: Float32Array, label: string) {
    for (let wire = 0; wire < values.length; wire++) {
      const range = machine.circuit.ranges[wire]!
      const value = values[wire]!
      const slack = 1e-4 * Math.max(
        1,
        Number.isFinite(range.minimum) ? Math.abs(range.minimum) : 0,
        Number.isFinite(range.maximum) ? Math.abs(range.maximum) : 0,
      )
      assert.ok(
        value >= range.minimum - slack && value <= range.maximum + slack,
        `${label}: wire ${wire} value ${value} escapes [${range.minimum}, ${range.maximum}]`,
      )
    }
  }

  const collision = new Float64Array(authoredFamilies()[0]!.fields)
  for (let slot = 0; slot < slotCount; slot++) {
    // Reversed order keys spaced inside the comparison band force colliding
    // ranks; maximal importance, height preference, and height coefficients
    // make the colliding position's raw height sum as large as the generative
    // ranges allow (two slots at 2,130px each).
    collision[fieldDimension(`slot${slot}.orderKey`)] = (slotCount - 1 - slot) * 1.3e-11
    collision[fieldDimension(`slot${slot}.importance`)] = 1
    collision[fieldDimension(`slot${slot}.heightPreference`)] = 1
  }
  collision[fieldDimension('baseHeight')] = 720
  collision[fieldDimension('heightFromImportance')] = 700
  const source = buildSource(collision, defaultContext)
  const values = machine.evaluateReference(source)
  const ranks = machine.output.ranks.map(wire => values[wire]!)
  assert.ok(new Set(ranks).size < slotCount, 'the scenario must actually collide')
  assertWiresWithinDeclaredRanges(values, 'rank collision')

  for (const {name, fields} of authoredFamilies()) {
    assertWiresWithinDeclaredRanges(
      machine.evaluateReference(buildSource(fields, defaultContext)),
      name,
    )
  }
})
