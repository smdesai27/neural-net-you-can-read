// Evaluate a genome at a viewport and read back what the page needs: tile
// rectangles, validity, and the upstream fitness terms the taste model uses.

import {type LayoutSnapshot, extractSnapshot, scoreFitness} from '../../vendor/handwired/lab/generative-fields/fitness.ts'
import type {FlowLayoutMachine} from './circuit.ts'
import {type Genome, type LayoutContext, buildExtendedSource, defaultContext, genes, slotCount, tileCount} from './schema.ts'

export type Tile = {x: number, y: number, width: number, height: number}

export type Layout = {
  source: Float32Array
  values: Float32Array
  tiles: Tile[]
  valid: boolean
  violationCount: number
  contentBottom: number
  // Cards per row, top to bottom: the layout's structure, e.g. [3, 4, 4, 1].
  rows: number[]
  snapshot: LayoutSnapshot
}

export function contextAt(viewportWidth: number, viewportHeight = defaultContext.viewportHeight): LayoutContext {
  return {...defaultContext, viewportWidth, viewportHeight}
}

export function readLayout(machine: FlowLayoutMachine, source: Float32Array, values: Float32Array): Layout {
  const {output} = machine
  const tiles: Tile[] = []
  for (let slot = 0; slot < tileCount; slot++) {
    const start = slot * 4
    tiles.push({
      x: values[output.slotGeometry[start]!]!,
      y: values[output.slotGeometry[start + 1]!]!,
      width: values[output.slotGeometry[start + 2]!]!,
      height: values[output.slotGeometry[start + 3]!]!,
    })
  }
  const snapshot = extractSnapshot(values, output, source)
  return {
    source,
    values,
    tiles,
    valid: values[output.validBit]! === 1,
    violationCount: values[output.violationCount]!,
    contentBottom: values[output.contentBottom]!,
    rows: rowStructure(machine, values),
    snapshot,
  }
}

export function evaluateLayout(machine: FlowLayoutMachine, genome: Genome, context: LayoutContext = defaultContext): Layout {
  const source = buildExtendedSource(genome, context)
  return readLayout(machine, source, machine.evaluate(source))
}

// Row lengths from the circuit's own wrap bits over the occupied rank positions.
export function rowStructure(machine: FlowLayoutMachine, values: Float32Array): number[] {
  const rows: number[] = []
  for (let position = 0; position < tileCount; position++) {
    if (position === 0 || values[machine.output.wrapBits[position]!]! === 1) rows.push(0)
    rows[rows.length - 1]!++
  }
  return rows
}

// The seven upstream fitness terms, each roughly 0..1, as the taste model's features.
export function tasteFeatures(layout: Layout): number[] {
  const {terms} = scoreFitness(layout.snapshot)
  return [terms.fill, terms.verticalFit, terms.importanceMatch, terms.balanceX, terms.sizeComfort, terms.rowEvenness, terms.contentRichness]
}

export const tasteFeatureNames = ['fill', 'vertical fit', 'importance match', 'balance', 'size comfort', 'row evenness', 'richness'] as const

// Seeded PRNG (mulberry32), so breeding and tests are reproducible.
export function createRandom(seed: number) {
  let state = seed >>> 0
  const next = () => {
    state = (state + 0x6D2B79F5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const gaussian = () => {
    const u = Math.max(next(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next())
  }
  return {next, gaussian}
}
export type Random = ReturnType<typeof createRandom>

// A uniformly random genome inside every gene's range.
export function sampleGenome(random: Random): Genome {
  const genome = new Float64Array(genes.length)
  for (let index = 0; index < genes.length; index++) {
    const gene = genes[index]!
    genome[index] = gene.minimum + (gene.maximum - gene.minimum) * random.next()
  }
  return genome
}

export {slotCount, tileCount}
