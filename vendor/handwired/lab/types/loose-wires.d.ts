// Lab folders written before wires had a type of their own include this file in
// their tsconfig, so plain numbers still type-check as wires in them, e.g.
// circuit.assumeRange(0, 0, 1) for source cell 0. New code gets wires from
// circuit.source(index), circuit.constant(value) or asWire(position) instead.
declare module '../../src/runtime/numeric-ui-core.ts' {
  interface WireTypeOptions {
    looseWires: true
  }
}

export {}
