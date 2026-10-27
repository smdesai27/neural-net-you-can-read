// Names for the circuit's neurons. The builder in circuit.ts tags the wires that
// mean something on their own (a tile's width, a wrap decision, an overlap
// check); completeLabels() gives every other operation the label of the tagged
// wire it ultimately feeds, so each of the circuit's operations gets a name.

import type {Circuit, Wire} from '../../vendor/handwired/src/runtime/numeric-ui-core.ts'

// The circuit's five stages, in the order data moves through them.
export const stages = ['order', 'size', 'flow', 'placement', 'checks'] as const
export type Stage = typeof stages[number]

export const stageNames: Record<Stage, string> = {
  order: 'Sort tiles into reading order',
  size: 'Size each tile',
  flow: 'Break tiles into rows',
  placement: 'Route results back to tiles',
  checks: 'Check the layout',
}

export type WireTag = {stage: Stage, label: string, slot?: number}
export type WireTags = Map<Wire, WireTag>

export function createTagger() {
  const tags: WireTags = new Map()
  return {
    tags,
    tag(wire: Wire, stage: Stage, label: string, slot?: number) {
      tags.set(wire, slot === undefined ? {stage, label} : {stage, label, slot})
    },
  }
}

export type NeuronLabel = WireTag & {
  // True when the builder tagged this wire itself; false when the label is
  // inherited from the tagged wire it feeds.
  named: boolean
  // The tagged wire this operation serves (itself when named).
  owner: Wire
}

// Every wire a term or gate of an operation reads.
export function readsOf(circuit: Circuit, operationIndex: number): Wire[] {
  const operation = circuit.operations[operationIndex]!
  return [...operation.leftTerms, ...operation.gateTerms].map(([wire]) => wire)
}

// Labels for every operation, indexed by operation (wire - sourceSize). Walks
// the operations from last to first, so a reader's label exists before the
// operations it reads are visited. Operations nothing reads are 'unused'.
export function completeLabels(circuit: Circuit, tags: WireTags, live: Uint8Array): (NeuronLabel | null)[] {
  const {sourceSize, operations} = circuit
  const labels: (NeuronLabel | null)[] = Array.from({length: operations.length}, () => null)
  const readers: number[][] = Array.from({length: operations.length}, () => [])
  for (let index = 0; index < operations.length; index++) {
    for (const wire of readsOf(circuit, index)) {
      if (wire >= sourceSize) readers[wire - sourceSize]!.push(index)
    }
  }
  for (let index = operations.length - 1; index >= 0; index--) {
    if (!live[index]) continue
    const wire = (sourceSize + index) as Wire
    const tag = tags.get(wire)
    if (tag) {
      labels[index] = {...tag, named: true, owner: wire}
      continue
    }
    const reader = readers[index]!.map(at => labels[at]).find(label => label !== null)
    if (reader) labels[index] = {...reader, named: false}
  }
  return labels
}

// Operations that some output depends on.
export function liveOperations(circuit: Circuit, outputs: readonly Wire[]): Uint8Array {
  const {sourceSize, operations} = circuit
  const live = new Uint8Array(operations.length)
  const stack = outputs.filter(wire => wire >= sourceSize).map(wire => wire - sourceSize)
  while (stack.length > 0) {
    const index = stack.pop()!
    if (live[index]) continue
    live[index] = 1
    for (const wire of readsOf(circuit, index)) {
      if (wire >= sourceSize && !live[wire - sourceSize]) stack.push(wire - sourceSize)
    }
  }
  return live
}
