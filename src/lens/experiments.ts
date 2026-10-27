// The interpretability experiments the lessons run: seams, ablation against
// gradients, activation patching, and probes. Each works on the real circuit
// through network.evaluate, so every number on the page is the network's own.

import type {Wire} from '../../vendor/handwired/src/runtime/numeric-ui-core.ts'
import {type NeuronLabel, type Stage} from '../layout/labels.ts'
import {contextAt} from '../layout/run.ts'
import {type Genome, buildExtendedSource, tileCount} from '../layout/schema.ts'
import type {Network} from './network.ts'

// ---------------------------------------------------------------- geometry

export function geometryOf(network: Network, values: Float32Array): Float64Array {
  const geometry = new Float64Array(tileCount * 4)
  for (let index = 0; index < geometry.length; index++) geometry[index] = values[network.output.slotGeometry[index]!]!
  return geometry
}

// Total pixels the tiles moved or resized between two layouts.
export function geometryDistance(a: Float64Array, b: Float64Array): number {
  let distance = 0
  for (let index = 0; index < a.length; index++) distance += Math.abs(a[index]! - b[index]!)
  return distance
}

// ---------------------------------------------------------------- seams

export type Seam = {
  // Where the row structure changes, to a hundredth of a pixel.
  width: number
  rowsBefore: number[]
  rowsAfter: number[]
  // The row-break neurons that flip there.
  flips: {position: number, wire: Wire, label: string, before: number, after: number}[]
}

function rowsOf(bits: number[]): number[] {
  const rows: number[] = []
  bits.forEach((bit, position) => { if (position === 0 || bit === 1) rows.push(0); rows[rows.length - 1]!++ })
  return rows
}

// Every width in [from, to] where a row-break neuron flips, found by scanning
// whole pixels and bisecting each change to a hundredth of a pixel.
export function findSeams(network: Network, genome: Genome, from: number, to: number, height: number): Seam[] {
  const wraps = network.output.wrapBits.slice(0, tileCount)
  const bitsAt = (width: number) => {
    const values = network.machine.evaluate(buildExtendedSource(genome, contextAt(width, height)))
    return wraps.map(wire => values[wire]!)
  }
  const same = (a: number[], b: number[]) => a.every((bit, index) => bit === b[index])
  const seams: Seam[] = []
  let previous = bitsAt(from)
  for (let width = from + 1; width <= to; width++) {
    const current = bitsAt(width)
    if (!same(current, previous)) {
      let low = width - 1
      let high = width
      for (let step = 0; step < 7; step++) {
        const middle = (low + high) / 2
        if (same(bitsAt(middle), previous)) low = middle
        else high = middle
      }
      const after = bitsAt(high)
      const flips = wraps.flatMap((wire, position) => after[position] === previous[position] ? [] : [{
        position, wire, label: network.label(wire)!.label, before: previous[position]!, after: after[position]!,
      }])
      seams.push({width: high, rowsBefore: rowsOf(previous), rowsAfter: rowsOf(after), flips})
      previous = after
      if (!same(current, after)) width--
      continue
    }
    previous = current
  }
  return seams
}

// One neuron's value across a range of widths, for plotting its kink.
export function traceAcrossWidths(network: Network, genome: Genome, wire: Wire, from: number, to: number, height: number, samples = 240) {
  const points: {width: number, value: number}[] = []
  for (let index = 0; index < samples; index++) {
    const width = from + (to - from) * index / (samples - 1)
    const values = network.machine.evaluate(buildExtendedSource(genome, contextAt(width, height)))
    points.push({width, value: values[wire]!})
  }
  return points
}

// ---------------------------------------------------------------- gradients

// The true derivative of `objective . values` with respect to every wire. Like
// the runtime's backpropagateReference, except a boolean atom passes back zero:
// it is a step function, flat on both sides of its threshold.
export function backpropagate(network: Network, values: Float32Array, outputGradients: Float64Array): Float64Array {
  const {circuit} = network
  const gradients = new Float64Array(outputGradients)
  const add = (terms: readonly (readonly [Wire, number])[], scale: number) => {
    for (const [wire, coefficient] of terms) gradients[wire]! += coefficient * scale
  }
  const termsValue = (terms: readonly (readonly [Wire, number])[], bias: number) => {
    let value = bias
    for (const [wire, coefficient] of terms) value += coefficient * values[wire]!
    return value
  }
  for (let index = circuit.operations.length - 1; index >= 0; index--) {
    const operation = circuit.operations[index]!
    const gradient = gradients[circuit.sourceSize + index]!
    if (gradient === 0) continue
    const macro = operation.macro
    if (macro?.kind === 'multiply') {
      gradients[macro.left]! += gradient * values[macro.right]!
      gradients[macro.right]! += gradient * values[macro.left]!
      continue
    }
    if (macro?.kind === 'select') {
      const gate = values[macro.gate]!
      gradients[macro.off]! += gradient * (1 - gate)
      gradients[macro.on]! += gradient * gate
      gradients[macro.gate]! += gradient * (values[macro.on]! - values[macro.off]!)
      continue
    }
    if (operation.kind === 'boolean') continue
    if (operation.kind === 'affine') {
      add(operation.leftTerms, gradient)
      continue
    }
    const left = termsValue(operation.leftTerms, operation.leftBias)
    const gate = termsValue(operation.gateTerms, operation.gateBias)
    add(operation.leftTerms, gradient * Math.max(0, gate))
    if (gate > 0) add(operation.gateTerms, gradient * left)
  }
  return gradients
}

export type Importance = {
  wire: Wire
  label: string
  stage: Stage
  isDecision: boolean
  // Its only readers are decisions, so its gradient is exactly zero.
  feedsDecision: boolean
  // Pixels the tiles move when this neuron is knocked out.
  ablation: number
  // |gradient x value| of the total geometry: what a saliency map would report.
  gradient: number
}

const describe = (label: NeuronLabel) => label.named ? label.label : `helper for "${label.label}"`

// Knock out each named neuron, and each neuron that only feeds a decision, and
// compare with gradient x input.
// Decisions (0/1 neurons) are flipped; other neurons are set to zero.
export function ablationVersusGradient(network: Network, source: Float32Array): Importance[] {
  const values = network.evaluate(source)
  const base = geometryOf(network, values)
  const objective = new Float64Array(values.length)
  for (const wire of network.output.slotGeometry.slice(0, tileCount * 4)) objective[wire] = 1
  const gradients = backpropagate(network, values, objective)
  const results: Importance[] = []
  for (let index = 0; index < network.labels.length; index++) {
    const label = network.labels[index]
    const feedsDecision = network.feedsDecision[index] === 1
    if (!label || (!label.named && !feedsDecision)) continue
    const wire = network.wireOf(index)
    const value = values[wire]!
    const isDecision = network.isDecision[index] === 1
    const knocked = isDecision ? 1 - value : 0
    if (knocked === value) {
      results.push({wire, label: describe(label), stage: label.stage, isDecision, feedsDecision, ablation: 0, gradient: Math.abs(gradients[wire]! * value)})
      continue
    }
    const after = geometryOf(network, network.evaluate(source, new Map([[wire, knocked]])))
    results.push({wire, label: describe(label), stage: label.stage, isDecision, feedsDecision, ablation: geometryDistance(base, after), gradient: Math.abs(gradients[wire]! * value)})
  }
  return results
}

// ---------------------------------------------------------------- patching

export type PatchResult = {
  // Fraction of the way from A's layout to B's that patching moved it: 0 = still
  // A, 1 = exactly B.
  effect: number
}

function patchEffect(network: Network, sourceA: Float32Array, valuesB: Float32Array, wires: Iterable<Wire>, baseA: Float64Array, baseB: Float64Array): number {
  const overrides = new Map<Wire, number>()
  for (const wire of wires) overrides.set(wire, valuesB[wire]!)
  const patched = geometryOf(network, network.evaluate(sourceA, overrides))
  const total = geometryDistance(baseA, baseB)
  return total === 0 ? 0 : 1 - geometryDistance(patched, baseB) / total
}

// Run A, but with one layer's neurons copied from run B, for every layer.
export function patchEachLayer(network: Network, sourceA: Float32Array, sourceB: Float32Array): number[] {
  const valuesA = network.evaluate(sourceA)
  const valuesB = network.evaluate(sourceB)
  const baseA = geometryOf(network, valuesA)
  const baseB = geometryOf(network, valuesB)
  return network.layers.map(layer => patchEffect(network, sourceA, valuesB, layer.map(network.wireOf), baseA, baseB))
}

// Run A with one named neuron copied from run B, for every named neuron whose
// value differs between the runs. Sorted by effect, largest first.
export function patchEachNamedNeuron(network: Network, sourceA: Float32Array, sourceB: Float32Array) {
  const valuesA = network.evaluate(sourceA)
  const valuesB = network.evaluate(sourceB)
  const baseA = geometryOf(network, valuesA)
  const baseB = geometryOf(network, valuesB)
  const results: {wire: Wire, label: string, stage: Stage, layer: number, effect: number}[] = []
  network.labels.forEach((label, index) => {
    if (!label?.named) return
    const wire = network.wireOf(index)
    if (valuesA[wire] === valuesB[wire]) return
    results.push({wire, label: label.label, stage: label.stage, layer: network.layerOf[index]!, effect: patchEffect(network, sourceA, valuesB, [wire], baseA, baseB)})
  })
  return results.sort((a, b) => b.effect - a.effect)
}

// Run A with every neuron of one stage copied from run B.
export function patchEachStage(network: Network, sourceA: Float32Array, sourceB: Float32Array, stages: readonly Stage[]): Map<Stage, number> {
  const valuesA = network.evaluate(sourceA)
  const valuesB = network.evaluate(sourceB)
  const baseA = geometryOf(network, valuesA)
  const baseB = geometryOf(network, valuesB)
  const result = new Map<Stage, number>()
  for (const stage of stages) {
    const wires: Wire[] = []
    network.labels.forEach((label, index) => { if (label?.stage === stage) wires.push(network.wireOf(index)) })
    result.set(stage, patchEffect(network, sourceA, valuesB, wires, baseA, baseB))
  }
  return result
}

// ---------------------------------------------------------------- probes

export type ProbeData = {sources: Float32Array[], values: Float32Array[], targets: number[]}

export type ProbeLayer = {
  layer: number
  // Held-out R^2 of a ridge regression from this layer's neurons to the target.
  r2: number
  features: number[]
  weights: Float64Array
  mean: Float64Array
  // Each feature's standard deviation on the training samples.
  scale: Float64Array
  bias: number
}

// Ridge regression from up to `maxFeatures` of a layer's highest-variance
// neurons to the target, trained on 80% of the samples, scored on the rest.
export function probeLayer(network: Network, data: ProbeData, layer: number, maxFeatures = 96, ridge = 1e-2): ProbeLayer {
  const operations = network.layers[layer]!
  const count = data.values.length
  const split = Math.floor(count * 0.8)
  const columns = operations.map(network.wireOf)
  const variance = columns.map(wire => {
    let mean = 0
    let square = 0
    for (const values of data.values) { mean += values[wire]!; square += values[wire]! ** 2 }
    mean /= count
    return square / count - mean * mean
  })
  const features = columns
    .map((wire, index) => ({wire, variance: variance[index]!}))
    .filter(entry => entry.variance > 1e-9)
    .sort((a, b) => b.variance - a.variance)
    .slice(0, maxFeatures)
    .map(entry => entry.wire)
  const size = features.length
  const mean = new Float64Array(size)
  const scale = new Float64Array(size)
  features.forEach((wire, index) => {
    let total = 0
    let square = 0
    for (let sample = 0; sample < split; sample++) { total += data.values[sample]![wire]!; square += data.values[sample]![wire]! ** 2 }
    mean[index] = total / split
    scale[index] = Math.sqrt(Math.max(1e-12, square / split - mean[index]! ** 2))
  })
  let targetMean = 0
  for (let sample = 0; sample < split; sample++) targetMean += data.targets[sample]!
  targetMean /= split
  const row = (sample: number) => features.map((wire, index) => (data.values[sample]![wire]! - mean[index]!) / scale[index]!)
  const gram = new Float64Array(size * size)
  const moment = new Float64Array(size)
  for (let sample = 0; sample < split; sample++) {
    const x = row(sample)
    const y = data.targets[sample]! - targetMean
    for (let i = 0; i < size; i++) {
      moment[i]! += x[i]! * y
      for (let j = 0; j <= i; j++) gram[i * size + j]! += x[i]! * x[j]!
    }
  }
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < i; j++) gram[j * size + i] = gram[i * size + j]!
    gram[i * size + i]! += ridge * split
  }
  const standardWeights = solveSymmetric(gram, moment, size)
  let residual = 0
  let spread = 0
  let heldMean = 0
  for (let sample = split; sample < count; sample++) heldMean += data.targets[sample]!
  heldMean /= Math.max(1, count - split)
  for (let sample = split; sample < count; sample++) {
    const x = row(sample)
    let prediction = targetMean
    for (let i = 0; i < size; i++) prediction += standardWeights[i]! * x[i]!
    residual += (data.targets[sample]! - prediction) ** 2
    spread += (data.targets[sample]! - heldMean) ** 2
  }
  const weights = new Float64Array(size)
  for (let i = 0; i < size; i++) weights[i] = standardWeights[i]! / scale[i]!
  return {layer, r2: size === 0 || spread === 0 ? 0 : Math.max(0, 1 - residual / spread), features, weights, mean, scale, bias: targetMean}
}

// Cholesky solve of (symmetric positive definite) A x = b.
function solveSymmetric(matrix: Float64Array, vector: Float64Array, size: number): Float64Array {
  const lower = new Float64Array(size * size)
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i * size + j]!
      for (let k = 0; k < j; k++) sum -= lower[i * size + k]! * lower[j * size + k]!
      lower[i * size + j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / lower[j * size + j]!
    }
  }
  const y = new Float64Array(size)
  for (let i = 0; i < size; i++) {
    let sum = vector[i]!
    for (let k = 0; k < i; k++) sum -= lower[i * size + k]! * y[k]!
    y[i] = sum / lower[i * size + i]!
  }
  const x = new Float64Array(size)
  for (let i = size - 1; i >= 0; i--) {
    let sum = y[i]!
    for (let k = i + 1; k < size; k++) sum -= lower[k * size + i]! * x[k]!
    x[i] = sum / lower[i * size + i]!
  }
  return x
}

// Amnesic test: remove the probe's direction from its layer in every held-out
// sample, re-run the network, and report how far the tiles moved and how well
// the same probe still reads the target from the edited layer.
export function eraseProbeDirection(network: Network, data: ProbeData, probe: ProbeLayer) {
  const norm = Math.sqrt(probe.weights.reduce((sum, weight) => sum + weight * weight, 0))
  const unit = probe.weights.map(weight => weight / (norm || 1))
  const split = Math.floor(data.values.length * 0.8)
  let moved = 0
  let residual = 0
  let spread = 0
  let heldMean = 0
  for (let sample = split; sample < data.values.length; sample++) heldMean += data.targets[sample]!
  heldMean /= Math.max(1, data.values.length - split)
  for (let sample = split; sample < data.values.length; sample++) {
    const values = data.values[sample]!
    let along = 0
    probe.features.forEach((wire, index) => { along += (values[wire]! - probe.mean[index]!) * unit[index]! })
    const overrides = new Map<Wire, number>()
    probe.features.forEach((wire, index) => overrides.set(wire, values[wire]! - along * unit[index]!))
    const edited = network.evaluate(data.sources[sample]!, overrides)
    moved += geometryDistance(geometryOf(network, values), geometryOf(network, edited))
    let prediction = probe.bias
    probe.features.forEach((wire, index) => { prediction += probe.weights[index]! * (edited[wire]! - probe.mean[index]!) })
    residual += (data.targets[sample]! - prediction) ** 2
    spread += (data.targets[sample]! - heldMean) ** 2
  }
  const held = Math.max(1, data.values.length - split)
  return {pixelsMovedPerLayout: moved / held, r2After: spread === 0 ? 0 : Math.max(0, 1 - residual / spread)}
}
