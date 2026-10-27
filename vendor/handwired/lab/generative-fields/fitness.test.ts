import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {authoredFamilies, buildSource, defaultContext, fieldDimension, globalField, slotCount} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, selectionScore, type LayoutSnapshot} from './fitness.ts'
import {createRandom, sampleFieldVector} from './search.ts'
import {seedFitnessGradients} from './optimize.ts'
import {backpropagateStraightThrough} from './ste-backprop.ts'

const machine = createFlowLayoutMachine()

function snapshotFrom(fields: Float64Array): LayoutSnapshot {
  const source = buildSource(fields, defaultContext)
  const values = machine.evaluate(source)
  return extractSnapshot(values, machine.output, source)
}

function cloneSnapshot(snapshot: LayoutSnapshot): LayoutSnapshot {
  return {
    ...snapshot,
    x: new Float64Array(snapshot.x),
    y: new Float64Array(snapshot.y),
    width: new Float64Array(snapshot.width),
    height: new Float64Array(snapshot.height),
    visible: [...snapshot.visible],
    importance: new Float64Array(snapshot.importance),
  }
}

// The fitness gradients treat geometry entries, coveredArea, and contentBottom
// as independent inputs (each seeds its own circuit wire), so the finite
// difference check perturbs them independently too.
test('hand-derived fitness gradients match central finite differences', () => {
  const random = createRandom(0xF17_0001)
  const snapshots = [
    ...authoredFamilies().map(({fields}) => snapshotFrom(fields)),
    ...Array.from({length: 12}, () => snapshotFrom(sampleFieldVector(random))),
  ]
  let checkedCount = 0
  let kinkSkips = 0
  for (const snapshot of snapshots) {
    const {gradient} = scoreFitness(snapshot)

    function check(
      label: string,
      analytic: number,
      applyOffset: (target: LayoutSnapshot, offset: number) => void,
      epsilon: number,
    ) {
      const above = cloneSnapshot(snapshot)
      applyOffset(above, epsilon)
      const below = cloneSnapshot(snapshot)
      applyOffset(below, -epsilon)
      const valueAbove = scoreFitness(above).value
      const valueBelow = scoreFitness(below).value
      const valueCenter = scoreFitness(snapshot).value
      const forwardSlope = (valueAbove - valueCenter) / epsilon
      const backwardSlope = (valueCenter - valueBelow) / epsilon
      // A kink (abs/min/max switching sides) inside the window makes the
      // one-sided slopes disagree; the analytic subgradient is still one of
      // them, so skip the comparison there.
      if (Math.abs(forwardSlope - backwardSlope) > 1e-7 + 0.02 * Math.abs(forwardSlope)) {
        kinkSkips++
        return
      }
      const finiteDifference = (valueAbove - valueBelow) / (2 * epsilon)
      assert.ok(
        Math.abs(analytic - finiteDifference) < 1e-9 + 2e-3 * Math.abs(finiteDifference),
        `${label}: analytic ${analytic} vs finite difference ${finiteDifference}`,
      )
      checkedCount++
    }

    for (let slot = 0; slot < slotCount; slot++) {
      if (!snapshot.visible[slot]!) continue
      check(`x[${slot}]`, gradient.x[slot]!, (target, offset) => {
        target.x[slot] = target.x[slot]! + offset
      }, 1e-3)
      check(`width[${slot}]`, gradient.width[slot]!, (target, offset) => {
        target.width[slot] = target.width[slot]! + offset
      }, 1e-3)
      check(`height[${slot}]`, gradient.height[slot]!, (target, offset) => {
        target.height[slot] = target.height[slot]! + offset
      }, 1e-3)
      check(`importance[${slot}]`, gradient.importance[slot]!, (target, offset) => {
        target.importance[slot] = target.importance[slot]! + offset
      }, 1e-5)
    }
    check('coveredArea', gradient.coveredArea, (target, offset) => {
      target.coveredArea += offset
    }, 1)
    check('contentBottom', gradient.contentBottom, (target, offset) => {
      target.contentBottom += offset
    }, 1e-3)
    for (const field of ['innerWidth', 'innerHeight', 'viewportWidth', 'viewportHeight', 'paddingY'] as const) {
      check(field, gradient[field], (target, offset) => {
        target[field] += offset
      }, 1e-3)
    }
  }
  assert.ok(checkedCount > 300, `too few gradient checks ran: ${checkedCount}`)
  assert.ok(kinkSkips < checkedCount * 0.2, `too many kink skips: ${kinkSkips} vs ${checkedCount}`)
})

test('fill gradient includes both geometry and direct padding dependencies', () => {
  const fields = authoredFamilies()[0]!.fields
  const source = buildSource(fields, defaultContext)
  const values = machine.evaluate(source)
  const snapshot = extractSnapshot(values, machine.output, source)
  const weights = {fill: 1, verticalFit: 0, importanceMatch: 0, balanceX: 0, sizeComfort: 0, rowEvenness: 0, contentRichness: 0}
  const gradient = scoreFitness(snapshot, weights).gradient
  const seeds = seedFitnessGradients(machine, gradient, snapshot, 1)
  const actual = backpropagateStraightThrough(machine.circuit, values, seeds)[globalField.paddingX]!
  const epsilon = 0.25
  const above = new Float64Array(fields)
  const below = new Float64Array(fields)
  above[fieldDimension('paddingX')]! += epsilon
  below[fieldDimension('paddingX')]! -= epsilon
  const expected = (scoreFitness(snapshotFrom(above), weights).value - scoreFitness(snapshotFrom(below), weights).value) / (2 * epsilon)
  assert.ok(Math.abs(actual - expected) < 1e-6, `complete derivative ${actual}, finite difference ${expected}`)
})

test('selection score ranks every valid layout above every invalid one', () => {
  const random = createRandom(0x5C0_4E00)
  const scores: {valid: boolean, score: number}[] = []
  for (let trial = 0; trial < 200; trial++) {
    const snapshot = snapshotFrom(sampleFieldVector(random))
    const fitness = scoreFitness(snapshot)
    assert.ok(fitness.value >= 0 && fitness.value <= 1)
    scores.push({valid: snapshot.valid, score: selectionScore(snapshot, fitness.value)})
  }
  const validScores = scores.filter(entry => entry.valid).map(entry => entry.score)
  const invalidScores = scores.filter(entry => !entry.valid).map(entry => entry.score)
  assert.ok(validScores.length > 20 && invalidScores.length > 20)
  assert.ok(Math.min(...validScores) > Math.max(...invalidScores))
})

test('content richness penalizes slab layouts that fill the viewport with few slots', () => {
  const grid = snapshotFrom(authoredFamilies()[0]!.fields)
  const gridFitness = scoreFitness(grid)
  assert.equal(gridFitness.terms.contentRichness, 1)

  // A synthetic two-slab layout: perfect fill, terrible richness.
  const slab = cloneSnapshot(grid)
  for (let slot = 0; slot < slotCount; slot++) slab.visible[slot] = slot < 2
  const slabFitness = scoreFitness(slab)
  assert.equal(slabFitness.terms.contentRichness, 0.2)
})
