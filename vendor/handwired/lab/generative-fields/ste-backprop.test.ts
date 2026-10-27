import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {Circuit, emitMultiply, emitRelu} from '../../src/runtime/numeric-ui-core.ts'
import {backpropagateStraightThrough} from './ste-backprop.ts'
import {authoredFamilies, buildSource, defaultContext, fieldDimension, generativeFields} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'

test('matches the reference backward pass when no boolean atom carries gradient', () => {
  const circuit = new Circuit(4)
  circuit.assumeRange(0, -2, 2)
  circuit.assumeRange(1, 0, 1)
  circuit.assumeRange(2, -1, 1)
  circuit.assumeRange(3, 0, 3)
  const product = emitMultiply(circuit, 0, 1)
  const rectified = emitRelu(circuit, circuit.affine([[product, 1], [2, -1]], 0.25))
  const output = circuit.affine([[rectified, 2], [3, 0.5]], -1)

  const source = Float32Array.from([1.5, 0.75, -0.5, 2])
  const values = circuit.evaluateReference(source)
  const outputGradients = new Float32Array(values.length)
  outputGradients[output] = 1
  const reference = circuit.backpropagateReference(values, outputGradients)
  const straightThrough = backpropagateStraightThrough(circuit, values, outputGradients)
  assert.deepEqual([...straightThrough], [...reference])
})

test('passes gradient through a boolean atom where the reference pass throws', () => {
  const circuit = new Circuit(1)
  circuit.assumeRange(0, 0, 1)
  const bit = circuit.boolean(0)
  const output = circuit.affine([[bit, 3]], 1)

  const source = Float32Array.from([0.75])
  const values = circuit.evaluateReference(source)
  const outputGradients = new Float32Array(values.length)
  outputGradients[output] = 1
  assert.throws(() => circuit.backpropagateReference(values, outputGradients))
  const gradients = backpropagateStraightThrough(circuit, values, outputGradients)
  assert.equal(gradients[0], 3)
})

// The estimator is exact on purely continuous paths (heights, gaps, padding
// flow into geometry through affine/reglu only). For fields that also feed
// discrete decisions (importance changes the span, and the span feeds the wrap
// booleans), the straight-through pass adds phantom gradient through the
// frozen decisions; there the test asserts sign and rough magnitude, which is
// the useful part for optimization. The README reports this bias.
test('field gradients through the layout circuit: exact on continuous paths, sign-correct near decisions', () => {
  const machine = createFlowLayoutMachine()
  const families = authoredFamilies()
  const probes: {family: number, field: string, delta: number, exact: boolean}[] = [
    {family: 0, field: 'baseHeight', delta: 0.25, exact: true},
    {family: 0, field: 'gapY', delta: 0.25, exact: true},
    {family: 0, field: 'paddingY', delta: 0.25, exact: true},
    {family: 1, field: 'baseHeight', delta: 0.25, exact: true},
    {family: 1, field: 'slot0.importance', delta: 0.002, exact: false},
    {family: 2, field: 'heightFromImportance', delta: 0.25, exact: false},
  ]
  for (const probe of probes) {
    const fields = families[probe.family]!.fields
    const dimension = fieldDimension(probe.field)

    function contentBottomAt(offset: number): number {
      const shifted = new Float64Array(fields)
      shifted[dimension] = shifted[dimension]! + offset
      const source = buildSource(shifted, defaultContext)
      return machine.evaluate(source)[machine.output.contentBottom]!
    }

    // Probes are chosen so no decision flips inside the probe window; verify
    // by comparing the two one-sided slopes.
    const centered = contentBottomAt(0)
    const above = contentBottomAt(probe.delta)
    const below = contentBottomAt(-probe.delta)
    const forwardSlope = (above - centered) / probe.delta
    const backwardSlope = (centered - below) / probe.delta
    assert.ok(
      Math.abs(forwardSlope - backwardSlope) < 0.05 * Math.max(1, Math.abs(forwardSlope)),
      `${probe.field}: a decision flipped inside the probe window`,
    )
    const finiteDifference = (above - below) / (2 * probe.delta)

    const source = buildSource(fields, defaultContext)
    const values = machine.evaluate(source)
    const outputGradients = new Float32Array(values.length)
    outputGradients[machine.output.contentBottom] = 1
    const gradients = backpropagateStraightThrough(machine.circuit, values, outputGradients)
    const analytic = gradients[generativeFields[dimension]!.source]!
    if (probe.exact) {
      assert.ok(
        Math.abs(analytic - finiteDifference)
          < 1e-3 + 0.02 * Math.abs(finiteDifference),
        `${probe.field}: straight-through ${analytic} vs finite difference ${finiteDifference}`,
      )
    } else {
      assert.ok(
        Math.sign(analytic) === Math.sign(finiteDifference)
          && Math.abs(analytic) < 3 * Math.abs(finiteDifference)
          && Math.abs(analytic) > Math.abs(finiteDifference) / 3,
        `${probe.field}: straight-through ${analytic} vs finite difference ${finiteDifference}`,
      )
    }
  }
})
