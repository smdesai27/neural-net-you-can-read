import type {Circuit} from '../../src/runtime/numeric-ui-core.ts'

export function backpropagateStraightThrough(
  circuit: Circuit,
  values: Float32Array,
  outputGradients: Float32Array,
): Float32Array {
  // Exact forward values, explicitly substituted identity derivative for booleans.
  return circuit.backpropagateReference(values, outputGradients, {booleanGradient: 'straight-through'})
}
