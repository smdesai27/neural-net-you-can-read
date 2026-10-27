import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {validateTensorEvaluation} from '../vendor/handwired/src/runtime/numeric-ui-core.ts'
import {evaluateOracle} from '../src/layout/oracle.ts'
import {contextAt, createRandom, evaluateLayout, sampleGenome} from '../src/layout/run.ts'
import {buildExtendedSource, seedGenome, slotCount, tileCount} from '../src/layout/schema.ts'
import {createNetwork} from '../src/lens/network.ts'

const network = createNetwork()
const {machine} = network

test('the seed genome is a valid layout at 1280 x 800', () => {
  const layout = evaluateLayout(machine, seedGenome(), contextAt(1280, 800))
  assert.equal(layout.valid, true, `violations: ${layout.violationCount}`)
  assert.equal(layout.rows.reduce((a, b) => a + b, 0), tileCount)
})

test('the minimum tile width changes the column count with the viewport width', () => {
  const narrow = evaluateLayout(machine, seedGenome(), contextAt(640, 2000))
  const wide = evaluateLayout(machine, seedGenome(), contextAt(1920, 2000))
  assert.ok(Math.max(...narrow.rows) < Math.max(...wide.rows), `narrow ${narrow.rows} wide ${wide.rows}`)
})

test('compiled tensor evaluation matches the reference interpreter on every live wire', () => {
  const random = createRandom(0x7E57_0001)
  for (let trial = 0; trial < 200; trial++) {
    const width = 360 + Math.floor(random.next() * 1561)
    const source = buildExtendedSource(sampleGenome(random), contextAt(width, 900))
    validateTensorEvaluation(source, machine.circuit, machine.evaluate(source))
  }
})

test('circuit agrees with the float64 oracle across random genomes and widths', () => {
  const random = createRandom(0x02AC_1E00)
  const trials = 400
  let knifeEdgeSkips = 0
  for (let trial = 0; trial < trials; trial++) {
    const width = 360 + Math.floor(random.next() * 1561)
    const source = buildExtendedSource(sampleGenome(random), contextAt(width, 900))
    const values = machine.evaluate(source)
    const oracle = evaluateOracle(source)
    const clearMargins = oracle.margins.orderKey > 1e-6 && oracle.margins.wrap > 1e-5 && oracle.margins.violation > 0.02
    if (!clearMargins) {
      knifeEdgeSkips++
      continue
    }
    for (let slot = 0; slot < slotCount; slot++) {
      const expected = oracle.slots[slot]!
      assert.equal(values[machine.output.ranks[slot]!], expected.rank)
      const start = slot * 4
      const actual = [0, 1, 2, 3].map(field => values[machine.output.slotGeometry[start + field]!]!)
      const wanted = [expected.x, expected.y, expected.width, expected.height]
      actual.forEach((value, field) => assert.ok(Math.abs(value - wanted[field]!) < 0.05, `slot ${slot} field ${field}: ${value} vs ${wanted[field]}`))
    }
    assert.equal(values[machine.output.containViolationCount], oracle.containViolationCount)
    assert.equal(values[machine.output.sizeViolationCount], oracle.sizeViolationCount)
    assert.equal(values[machine.output.overlapViolationCount], 0)
    assert.equal(values[machine.output.validBit], oracle.valid ? 1 : 0)
  }
  assert.ok(knifeEdgeSkips < trials * 0.1, `too many knife-edge skips: ${knifeEdgeSkips}/${trials}`)
})

test('the lens evaluator equals the reference interpreter bit for bit with no overrides', () => {
  const random = createRandom(0xB17_B17)
  for (let trial = 0; trial < 100; trial++) {
    const source = buildExtendedSource(sampleGenome(random), contextAt(360 + Math.floor(random.next() * 1561), 900))
    const reference = machine.circuit.evaluateReference(source)
    const lens = network.evaluate(source)
    assert.equal(lens.length, reference.length)
    for (let wire = 0; wire < lens.length; wire++) {
      if (!Object.is(lens[wire], reference[wire])) assert.fail(`wire ${wire}: ${lens[wire]} vs ${reference[wire]}`)
    }
  }
})

test('overriding a wrap decision moves tiles downstream of it', () => {
  const source = buildExtendedSource(seedGenome(), contextAt(1280, 800))
  const before = network.evaluate(source)
  const wrapAt = machine.output.wrapBits.findIndex((wire, position) => position > 0 && before[wire] === 1)
  assert.ok(wrapAt > 0)
  const after = network.evaluate(source, new Map([[machine.output.wrapBits[wrapAt]!, 0]]))
  const moved = machine.output.slotGeometry.filter(wire => before[wire] !== after[wire])
  assert.ok(moved.length > 0)
})

test('every live operation has a label, and the builder named the important ones', () => {
  let live = 0
  let named = 0
  for (let index = 0; index < network.labels.length; index++) {
    if (!network.live[index]) continue
    live++
    assert.ok(network.labels[index], `operation ${index} has no label`)
    if (network.labels[index]!.named) named++
  }
  assert.ok(live > 1000 && named > 300, `live ${live} named ${named}`)
  console.log(`live operations ${live}, named ${named}, layers ${network.layers.length}, built ${machine.circuit.operations.length}, compiled layers ${machine.tensorProgram.layers.length}`)
})
