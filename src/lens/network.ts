// The layout circuit seen as a network: which operations are neurons, which
// layer each sits in, what each one is called, and an evaluator that can
// overwrite any neuron's value (ablation, patching) while the rest of the
// network runs as usual.

import type {Circuit, Wire} from '../../vendor/handwired/src/runtime/numeric-ui-core.ts'
import {createFlowLayoutMachine, flowLayoutOutputWires} from '../layout/circuit.ts'
import {type NeuronLabel, completeLabels, liveOperations, readsOf} from '../layout/labels.ts'
import {tileCount} from '../layout/schema.ts'

export type Network = ReturnType<typeof createNetwork>

export function createNetwork() {
  const machine = createFlowLayoutMachine()
  const {circuit} = machine
  const outputs = flowLayoutOutputWires(machine.output)
  const live = liveOperations(circuit, outputs)
  // Slots past tileCount are hidden by the genome; say so wherever a label names one.
  for (const tag of machine.tags.values()) {
    tag.label = tag.label.replace(/\btiles? (\d+)/g, (text, slot: string) => Number(slot) > tileCount ? `hidden ${text}` : text)
      .replace(/\band (\d+)/g, (text, slot: string) => Number(slot) > tileCount ? `and hidden ${slot}` : text)
  }
  const labels = completeLabels(circuit, machine.tags, live)
  const {layerOf, layers} = layerOperations(circuit, live)
  const neuronCount = layers.reduce((sum, layer) => sum + layer.length, 0)
  const isDecision = Uint8Array.from(circuit.operations, operation => operation.kind === 'boolean' ? 1 : 0)
  // Neurons the tile geometry depends on only through decisions: every path
  // from them to a tile crosses a 0/1 step, so their gradient is exactly zero
  // whatever the input.
  const geometry = machine.output.slotGeometry.slice(0, tileCount * 4)
  const reachesAny = reachBack(circuit, geometry, () => true)
  const reachesSmoothly = reachBack(circuit, geometry, index => isDecision[index] === 0)
  const feedsDecision = Uint8Array.from(reachesAny, (reached, index) => reached && !reachesSmoothly[index] ? 1 : 0)
  return {
    isDecision,
    feedsDecision,
    machine,
    circuit,
    output: machine.output,
    sourceSize: circuit.sourceSize,
    live,
    labels,
    // layerOf[operation] is the operation's layer (0-based), or -1 when no output reads it.
    layerOf,
    // layers[layer] lists the operations in that layer, sorted by stage and tile.
    layers,
    neuronCount,
    wireOf: (operation: number) => (circuit.sourceSize + operation) as Wire,
    operationOf: (wire: Wire) => wire - circuit.sourceSize,
    label(wire: Wire): NeuronLabel | null {
      return wire >= circuit.sourceSize ? labels[wire - circuit.sourceSize] ?? null : null
    },
    evaluate(source: Float32Array, overrides?: ReadonlyMap<Wire, number>): Float32Array {
      return evaluateWithOverrides(circuit, source, overrides)
    },
  }
}

// Operations some wire in `roots` depends on, walking back through an
// operation's reads only when `through(operation)` allows it.
function reachBack(circuit: Circuit, roots: readonly Wire[], through: (operation: number) => boolean): Uint8Array {
  const reached = new Uint8Array(circuit.operations.length)
  const stack = roots.filter(wire => wire >= circuit.sourceSize).map(wire => wire - circuit.sourceSize)
  while (stack.length > 0) {
    const index = stack.pop()!
    if (reached[index]) continue
    reached[index] = 1
    if (!through(index)) continue
    for (const wire of readsOf(circuit, index)) {
      if (wire >= circuit.sourceSize && !reached[wire - circuit.sourceSize]) stack.push(wire - circuit.sourceSize)
    }
  }
  return reached
}

// A neuron's layer is one more than the deepest neuron it reads, sources being
// layer -1, so layer 0 reads only the source table.
function layerOperations(circuit: Circuit, live: Uint8Array) {
  const {sourceSize, operations} = circuit
  const layerOf = new Int32Array(operations.length).fill(-1)
  for (let index = 0; index < operations.length; index++) {
    if (!live[index]) continue
    let layer = 0
    for (const wire of readsOf(circuit, index)) {
      if (wire >= sourceSize) layer = Math.max(layer, layerOf[wire - sourceSize]! + 1)
    }
    layerOf[index] = layer
  }
  const layerCount = Math.max(0, ...layerOf) + 1
  const layers: number[][] = Array.from({length: layerCount}, () => [])
  for (let index = 0; index < operations.length; index++) {
    if (layerOf[index]! >= 0) layers[layerOf[index]!]!.push(index)
  }
  return {layerOf, layers}
}

// Mirrors the runtime's evaluateOperationStream (vendor/.../numeric-ui-core.ts),
// FP32 rounding included, with one addition: an overridden wire takes the given
// value instead of its computed one, and everything downstream reads that.
// With no overrides it must equal circuit.evaluateReference bit for bit, which
// tests/lens.test.ts checks.
export function evaluateWithOverrides(
  circuit: Circuit,
  source: Float32Array,
  overrides?: ReadonlyMap<Wire, number>,
): Float32Array {
  const {sourceSize, operations} = circuit
  if (source.length !== sourceSize) throw new Error('source table has the wrong length')
  const values = new Float32Array(sourceSize + operations.length)
  values.set(source)
  for (let index = 0; index < operations.length; index++) {
    const wire = (sourceSize + index) as Wire
    const override = overrides?.get(wire)
    if (override !== undefined) {
      values[wire] = override
      continue
    }
    const operation = operations[index]!
    let value = evaluateTerms(values, operation.leftTerms, operation.leftBias)
    if (operation.kind === 'reglu') {
      value = Math.fround(value * Math.max(0, evaluateTerms(values, operation.gateTerms, operation.gateBias)))
    } else if (operation.kind === 'boolean') {
      value = evaluateBoolean(value)
    } else if (operation.kind === 'index') {
      throw new Error('the layout circuit has no index reads')
    }
    values[wire] = value
  }
  return values
}

function evaluateTerms(values: Float32Array, terms: readonly (readonly [Wire, number])[], bias: number): number {
  let value = Math.fround(bias)
  for (const [wire, coefficient] of terms) {
    value = Math.fround(value + Math.fround(Math.fround(coefficient) * values[wire]!))
  }
  return value
}

// The runtime's boolean atom: FP32 rounds to a whole number at this size, and
// the odd base makes exactly 0.5 round up.
function evaluateBoolean(value: number): number {
  const base = 8_388_609
  return Math.fround(Math.fround(Math.fround(value) + base) - base)
}
