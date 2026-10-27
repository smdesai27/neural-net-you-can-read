// Gradient-based generation and validity projection over the field vector.
//
// Both entry points share one mechanism: evaluate the circuit, seed gradients
// on output wires (fitness gradients from fitness.ts, or a unit gradient on
// the violation depth wire), run the straight-through backward pass down to
// the source cells, and take Adam steps on the generative fields. Steps happen
// in normalized field space (every field mapped to 0..1 by its declared range)
// so one learning rate serves pixel fields and fraction fields alike; after
// every step the fields clamp back into their ranges.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'
import {
  buildSource,
  clampFields,
  generativeFieldCount,
  generativeFields,
  type LayoutContext,
} from './fields.ts'
import type {FlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, selectionScore, type FitnessGradient, type FitnessWeights, type LayoutSnapshot} from './fitness.ts'
import {geometryField} from './flow-layout.ts'
import {globalField, slotCount, slotField, slotFieldIndex} from './fields.ts'
import {backpropagateStraightThrough} from './ste-backprop.ts'

type AdamState = {
  firstMoment: Float64Array
  secondMoment: Float64Array
  step: number
}

function createAdamState(): AdamState {
  return {
    firstMoment: new Float64Array(generativeFieldCount),
    secondMoment: new Float64Array(generativeFieldCount),
    step: 0,
  }
}

// One Adam descent step in normalized field space. `lossGradients` holds
// dLoss/dField per generative dimension in raw field units.
function adamStep(
  fields: Float64Array,
  lossGradients: Float64Array,
  state: AdamState,
  learningRate: number,
) {
  const firstDecay = 0.9
  const secondDecay = 0.999
  state.step++
  const firstCorrection = 1 - firstDecay ** state.step
  const secondCorrection = 1 - secondDecay ** state.step
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    const descriptor = generativeFields[dimension]!
    const rangeWidth = descriptor.maximum - descriptor.minimum
    if (rangeWidth === 0) continue
    // Chain rule into normalized coordinates: field = min + normalized * width.
    const normalizedGradient = lossGradients[dimension]! * rangeWidth
    state.firstMoment[dimension] =
      firstDecay * state.firstMoment[dimension]! + (1 - firstDecay) * normalizedGradient
    state.secondMoment[dimension] =
      secondDecay * state.secondMoment[dimension]!
      + (1 - secondDecay) * normalizedGradient * normalizedGradient
    const correctedFirst = state.firstMoment[dimension]! / firstCorrection
    const correctedSecond = state.secondMoment[dimension]! / secondCorrection
    const normalizedDelta =
      learningRate * correctedFirst / (Math.sqrt(correctedSecond) + 1e-8)
    fields[dimension] = fields[dimension]! - normalizedDelta * rangeWidth
  }
  clampFields(fields)
}

// Backpropagate seeded output gradients to the generative field dimensions.
function fieldLossGradients(
  machine: FlowLayoutMachine,
  values: Float32Array,
  outputGradients: Float32Array,
): Float64Array {
  const sourceGradients = backpropagateStraightThrough(machine.circuit, values, outputGradients)
  const lossGradients = new Float64Array(generativeFieldCount)
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    lossGradients[dimension] = sourceGradients[generativeFields[dimension]!.source]!
  }
  return lossGradients
}

export function seedFitnessGradients(
  machine: FlowLayoutMachine,
  gradient: FitnessGradient,
  snapshot: LayoutSnapshot,
  sign: number,
): Float32Array {
  const width = machine.circuit.sourceSize + machine.circuit.operations.length
  const outputGradients = new Float32Array(width)
  const output = machine.output
  // Accumulate rather than assign: expression deduplication may alias two
  // output wires onto one circuit wire.
  function seed(wire: number, value: number) {
    outputGradients[wire] = Math.fround(outputGradients[wire]! + sign * value)
  }
  for (let slot = 0; slot < slotCount; slot++) {
    const start = slot * geometryField.size
    seed(output.slotGeometry[start + geometryField.x]!, gradient.x[slot]!)
    seed(output.slotGeometry[start + geometryField.y]!, gradient.y[slot]!)
    seed(output.slotGeometry[start + geometryField.width]!, gradient.width[slot]!)
    seed(output.slotGeometry[start + geometryField.height]!, gradient.height[slot]!)
    seed(output.visibleBits[slot]!, gradient.visible[slot]!)
    seed(slotFieldIndex(slot, slotField.importance), gradient.importance[slot]!)
  }
  seed(output.coveredArea, gradient.coveredArea)
  seed(output.contentBottom, gradient.contentBottom)
  seed(globalField.viewportWidth, gradient.viewportWidth)
  seed(globalField.viewportHeight, gradient.viewportHeight)
  seed(globalField.paddingY, gradient.paddingY)
  // extractSnapshot derives the inner dimensions directly from source cells.
  // These paths do not pass through the geometry output wires.
  if (snapshot.viewportWidth - 2 * snapshot.paddingX > 1) {
    seed(globalField.viewportWidth, gradient.innerWidth)
    seed(globalField.paddingX, -2 * gradient.innerWidth)
  }
  if (snapshot.viewportHeight - 2 * snapshot.paddingY > 1) {
    seed(globalField.viewportHeight, gradient.innerHeight)
    seed(globalField.paddingY, -2 * gradient.innerHeight)
  }
  return outputGradients
}

export type AscentOptions = {
  steps?: number
  learningRate?: number
  // Pressure toward validity while ascending fitness, in loss units per
  // pixel of violation depth.
  violationWeight?: number
  // Alternative fitness weighting; the reward-hacking characterization test
  // passes the pre-richness weighting here.
  weights?: FitnessWeights
}

export type AscentResult = {
  fields: Float64Array
  fitness: number
  valid: boolean
  score: number
  evaluations: number
}

// Gradient ascent on fitness from a starting field vector. Tracks the best
// visited point by selection score (valid layouts always beat invalid ones).
export function ascendFitness(
  machine: FlowLayoutMachine,
  context: LayoutContext,
  startFields: Float64Array,
  options: AscentOptions = {},
): AscentResult {
  const steps = options.steps ?? 150
  const learningRate = options.learningRate ?? 0.02
  const violationWeight = options.violationWeight ?? 0.002
  const fields = new Float64Array(startFields)
  clampFields(fields)
  const adam = createAdamState()
  let best: AscentResult | null = null
  let evaluations = 0
  for (let step = 0; step <= steps; step++) {
    const source = buildSource(fields, context)
    const values = machine.evaluate(source)
    evaluations++
    const snapshot = extractSnapshot(values, machine.output, source)
    const fitness = scoreFitness(snapshot, options.weights)
    const score = selectionScore(snapshot, fitness.value)
    if (best === null || score > best.score) {
      best = {
        fields: new Float64Array(fields),
        fitness: fitness.value,
        valid: snapshot.valid,
        score,
        evaluations,
      }
    }
    if (step === steps) break
    // Loss = -fitness + violationWeight * violationDepth; descend it.
    const outputGradients = seedFitnessGradients(machine, fitness.gradient, snapshot, -1)
    outputGradients[machine.output.violationDepth] = Math.fround(
      outputGradients[machine.output.violationDepth]! + violationWeight,
    )
    const lossGradients = fieldLossGradients(machine, values, outputGradients)
    adamStep(fields, lossGradients, adam, learningRate)
  }
  invariant(best !== null, 'ascent evaluated no points')
  best.evaluations = evaluations
  return best
}

export type ProjectionOptions = {
  maxSteps?: number
  learningRate?: number
  // Weight of the proximal pull back toward the anchor point, applied in
  // normalized field space so the projected point stays near the path.
  proximalWeight?: number
}

export type ProjectionResult = {
  fields: Float64Array
  valid: boolean
  steps: number
  // Euclidean distance moved from the anchor, in normalized field space.
  movedDistance: number
}

// Project an invalid field vector back to validity: descend the circuit's own
// continuous violation depth, with a proximal term keeping the point close to
// the anchor (the interpolated point being repaired). Stops as soon as the
// circuit's exact valid bit turns on.
export function projectToValidity(
  machine: FlowLayoutMachine,
  context: LayoutContext,
  startFields: Float64Array,
  options: ProjectionOptions = {},
): ProjectionResult {
  const maxSteps = options.maxSteps ?? 40
  const learningRate = options.learningRate ?? 0.02
  const proximalWeight = options.proximalWeight ?? 0.05
  const anchor = new Float64Array(startFields)
  const fields = new Float64Array(startFields)
  clampFields(fields)
  const adam = createAdamState()
  const width = machine.circuit.sourceSize + machine.circuit.operations.length
  for (let step = 0; step <= maxSteps; step++) {
    const source = buildSource(fields, context)
    const values = machine.evaluate(source)
    if (values[machine.output.validBit]! === 1) {
      return {fields, valid: true, steps: step, movedDistance: normalizedDistance(fields, anchor)}
    }
    if (step === maxSteps) break
    const outputGradients = new Float32Array(width)
    outputGradients[machine.output.violationDepth] = 1
    const lossGradients = fieldLossGradients(machine, values, outputGradients)
    // Proximal pull: d/dField of proximalWeight * ||normalized - anchor||^2.
    for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
      const descriptor = generativeFields[dimension]!
      const rangeWidth = descriptor.maximum - descriptor.minimum
      if (rangeWidth === 0) continue
      const normalizedOffset = (fields[dimension]! - anchor[dimension]!) / rangeWidth
      lossGradients[dimension] = lossGradients[dimension]!
        + 2 * proximalWeight * normalizedOffset / rangeWidth
    }
    adamStep(fields, lossGradients, adam, learningRate)
  }
  return {fields, valid: false, steps: maxSteps, movedDistance: normalizedDistance(fields, anchor)}
}

export function normalizedDistance(a: Float64Array, b: Float64Array): number {
  let sum = 0
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    const descriptor = generativeFields[dimension]!
    const rangeWidth = descriptor.maximum - descriptor.minimum
    if (rangeWidth === 0) continue
    const offset = (a[dimension]! - b[dimension]!) / rangeWidth
    sum += offset * offset
  }
  return Math.sqrt(sum)
}
