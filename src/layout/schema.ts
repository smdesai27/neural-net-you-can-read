// Source table and genome for our fork of the generative-fields layout circuit.
//
// The vendored schema (vendor/handwired/lab/generative-fields/fields.ts) owns the
// first `sourceSize` cells unchanged, so its snapshot and fitness code read our
// tables as they are. We append two cells after it:
// - minTilePx: a genome gene, the narrowest a tile may be in pixels.
// - invInnerWidth: 1 / innerWidth, written by the host. Division stays outside
//   the circuit, following the repo rule that the host side owns division.
//
// The genome is the vendored generative field vector minus the visibility
// fields (pinned: the first `tileCount` slots shown, the rest hidden), plus
// minTilePx.

import {
  buildSource,
  defaultContext,
  generativeFields,
  globalField,
  type LayoutContext,
  slotCount,
  slotField,
  slotFieldIndex,
  sourceRanges,
  sourceSize,
} from '../../vendor/handwired/lab/generative-fields/fields.ts'

export {defaultContext, globalField, slotCount, slotField, slotFieldIndex, type LayoutContext}

export const tileCount = 12

export const minTilePxCell = sourceSize
export const invInnerWidthCell = sourceSize + 1
export const extendedSourceSize = sourceSize + 2

export const minTilePxRange = [0, 480] as const
export const invInnerWidthRange = [1 / 8192, 1] as const

export const extendedSourceRanges: readonly (readonly [number, number])[] = [
  ...sourceRanges,
  minTilePxRange,
  invInnerWidthRange,
]

export type Gene = {name: string, minimum: number, maximum: number, cell: number}

const visibilityCells = new Set(
  Array.from({length: slotCount}, (_, slot) => slotFieldIndex(slot, slotField.visibility)),
)

// Genes over the hidden slots move nothing on screen, so they are left out too.
const hiddenSlotCells = new Set(
  Array.from({length: slotCount - tileCount}, (_, index) => index + tileCount)
    .flatMap(slot => [slotField.importance, slotField.orderKey, slotField.heightPreference]
      .map(field => slotFieldIndex(slot, field))),
)

export const genes: readonly Gene[] = [
  ...generativeFields
    .filter(field => !visibilityCells.has(field.source) && !hiddenSlotCells.has(field.source))
    .map(field => ({name: field.name, minimum: field.minimum, maximum: field.maximum, cell: field.source})),
  {name: 'minTilePx', minimum: minTilePxRange[0], maximum: minTilePxRange[1], cell: minTilePxCell},
]

export type Genome = Float64Array

export function geneIndex(name: string): number {
  const index = genes.findIndex(gene => gene.name === name)
  if (index < 0) throw new Error(`no gene named ${name}`)
  return index
}

export function clampGenome(genome: Genome): Genome {
  for (let index = 0; index < genes.length; index++) {
    const gene = genes[index]!
    genome[index] = Math.min(gene.maximum, Math.max(gene.minimum, genome[index]!))
  }
  return genome
}

// The circuit's innerWidth is relu(viewportWidth - 2 paddingX) in FP32; the host
// mirrors that rounding before taking the reciprocal.
export function inverseInnerWidth(viewportWidth: number, paddingX: number): number {
  const innerWidth = Math.fround(Math.fround(viewportWidth) - Math.fround(2 * Math.fround(paddingX)))
  return Math.fround(1 / Math.min(8192, Math.max(1, innerWidth)))
}

export function buildExtendedSource(genome: Genome, context: LayoutContext = defaultContext): Float32Array {
  if (genome.length !== genes.length) throw new Error('genome has the wrong length')
  const fields = new Float64Array(generativeFields.length)
  const cellToField = new Map(generativeFields.map((field, index) => [field.source, index]))
  for (let slot = 0; slot < slotCount; slot++) {
    fields[cellToField.get(slotFieldIndex(slot, slotField.visibility))!] = slot < tileCount ? 1 : 0
    if (slot >= tileCount) {
      fields[cellToField.get(slotFieldIndex(slot, slotField.importance))!] = 0
      fields[cellToField.get(slotFieldIndex(slot, slotField.orderKey))!] = 1
      fields[cellToField.get(slotFieldIndex(slot, slotField.heightPreference))!] = 0
    }
  }
  for (let index = 0; index < genes.length; index++) {
    const gene = genes[index]!
    const value = genome[index]!
    if (!(value >= gene.minimum && value <= gene.maximum)) {
      throw new Error(`gene ${gene.name} is outside its range: ${value}`)
    }
    const field = cellToField.get(gene.cell)
    if (field !== undefined) fields[field] = value
  }
  const source = new Float32Array(extendedSourceSize)
  source.set(buildSource(fields, context))
  source[minTilePxCell] = genome[genes.length - 1]!
  source[invInnerWidthCell] = inverseInnerWidth(context.viewportWidth, source[globalField.paddingX]!)
  return source
}

// A genome that reads as a plain grid: equal tiles, at least 240 px wide, in
// reading order. Breeding starts here.
export function seedGenome(): Genome {
  const genome = new Float64Array(genes.length)
  const set = (name: string, value: number) => { genome[geneIndex(name)] = value }
  set('paddingX', 32)
  set('paddingY', 32)
  set('gapX', 0.015)
  set('gapY', 16)
  set('baseSpan', 0.1)
  set('spanFromImportance', 0.35)
  set('baseHeight', 125)
  set('heightFromImportance', 145)
  set('minTilePx', 240)
  for (let slot = 0; slot < tileCount; slot++) {
    set(`slot${slot}.importance`, slot === 0 ? 0.7 : 0.15)
    set(`slot${slot}.orderKey`, (slot + 0.5) / slotCount)
    set(`slot${slot}.heightPreference`, 0.5)
  }
  return genome
}
