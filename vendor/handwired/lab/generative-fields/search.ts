// Generation by search: sample seeded field vectors, run the compiled circuit
// on each, score with the fitness function, keep the best. This is the Duplo
// recipe (thousands of candidates, weighted heuristics, pick the winner) with
// the layout and the validity checks living inside one circuit.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'
import {
  buildSource,
  fieldDimension,
  generativeFieldCount,
  type LayoutContext,
  slotCount,
  slotField,
} from './fields.ts'
import type {FlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, selectionScore, type LayoutSnapshot} from './fitness.ts'

// The repo's LCG pattern, seeded and deterministic.
export function createRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0
    return state / 0x1_0000_0000
  }
}

function uniform(random: () => number, minimum: number, maximum: number) {
  return minimum + (maximum - minimum) * random()
}

// Sampling ranges for the global knobs: a designer-plausible sub-box of the
// declared field ranges. Per-slot fields sample their full 0..1 ranges.
const samplerGlobals: readonly (readonly [name: string, minimum: number, maximum: number])[] = [
  ['paddingX', 8, 48],
  ['paddingY', 8, 48],
  ['gapX', 0.004, 0.03],
  ['gapY', 4, 24],
  ['baseSpan', 0.05, 0.55],
  ['spanFromImportance', 0, 0.6],
  ['baseHeight', 40, 260],
  ['heightFromImportance', 0, 220],
]

export function sampleFieldVector(random: () => number): Float64Array {
  const fields = new Float64Array(generativeFieldCount)
  for (const [name, minimum, maximum] of samplerGlobals) {
    fields[fieldDimension(name)] = uniform(random, minimum, maximum)
  }
  for (let slot = 0; slot < slotCount; slot++) {
    const base = samplerGlobals.length + slot * slotField.size
    fields[base + slotField.importance] = random()
    fields[base + slotField.orderKey] = random()
    fields[base + slotField.visibility] = random()
    fields[base + slotField.heightPreference] = random()
  }
  return fields
}

export type SearchCandidate = {
  fields: Float64Array
  snapshot: LayoutSnapshot
  fitness: number
  score: number
  sampleIndex: number
}

export type SearchResult = {
  sampleCount: number
  validCount: number
  // Samples with at least one violation, split by category. A sample can
  // count toward more than one category.
  overlapViolationSamples: number
  containViolationSamples: number
  sizeViolationSamples: number
  top: SearchCandidate[]
  // Fitness values of the valid samples and selection scores of all samples,
  // both sorted ascending, for distribution reporting.
  validFitness: number[]
  allScores: number[]
}

export function runSearch(
  machine: FlowLayoutMachine,
  context: LayoutContext,
  options: {sampleCount: number, seed: number, keep?: number},
): SearchResult {
  const {sampleCount, seed} = options
  const keep = options.keep ?? 10
  invariant(sampleCount > 0 && keep > 0, 'search needs positive counts')
  const random = createRandom(seed)
  const top: SearchCandidate[] = []
  const validFitness: number[] = []
  const allScores: number[] = []
  let validCount = 0
  let overlapViolationSamples = 0
  let containViolationSamples = 0
  let sizeViolationSamples = 0
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const fields = sampleFieldVector(random)
    const source = buildSource(fields, context)
    const values = machine.evaluate(source)
    const snapshot = extractSnapshot(values, machine.output, source)
    const fitness = scoreFitness(snapshot)
    const score = selectionScore(snapshot, fitness.value)
    allScores.push(score)
    if (snapshot.valid) {
      validCount++
      validFitness.push(fitness.value)
    }
    if (snapshot.overlapViolationCount > 0) overlapViolationSamples++
    if (snapshot.containViolationCount > 0) containViolationSamples++
    if (snapshot.sizeViolationCount > 0) sizeViolationSamples++
    if (top.length < keep || score > top.at(-1)!.score) {
      top.push({fields, snapshot, fitness: fitness.value, score, sampleIndex})
      top.sort((a, b) => b.score - a.score)
      if (top.length > keep) top.pop()
    }
  }
  validFitness.sort((a, b) => a - b)
  allScores.sort((a, b) => a - b)
  return {
    sampleCount,
    validCount,
    overlapViolationSamples,
    containViolationSamples,
    sizeViolationSamples,
    top,
    validFitness,
    allScores,
  }
}

export function percentile(sortedValues: readonly number[], fraction: number): number {
  invariant(sortedValues.length > 0, 'percentile needs values')
  const position = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.round(fraction * (sortedValues.length - 1))),
  )
  return sortedValues[position]!
}
