// Field schema for the generative flow layout experiment.
//
// Every quantity that makes one slot different from another is a per-slot input
// field, and every global layout knob is a global input field. The circuit in
// flow-layout.ts consumes these through one shared per-slot function; nothing
// in this experiment forks code per slot.
//
// The source table splits into three groups:
// - context: viewport size plus fixed thresholds (wrap slack, minimum visible
//   sizes). Held constant during generation and interpolation.
// - global knobs: padding, gaps, and the span/height coefficients that expand
//   importance into geometry. Part of the generative field vector.
// - per-slot fields: importance, order key, visibility, height preference.
//   Part of the generative field vector.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'

export const slotCount = 16

export const globalField = {
  viewportWidth: 0,
  viewportHeight: 1,
  paddingX: 2,
  paddingY: 3,
  gapX: 4,
  gapY: 5,
  baseSpan: 6,
  spanFromImportance: 7,
  baseHeight: 8,
  heightFromImportance: 9,
  wrapSlack: 10,
  minVisibleWidth: 11,
  minVisibleHeight: 12,
  unit: 13,
  size: 14,
} as const

export const slotField = {
  importance: 0,
  orderKey: 1,
  visibility: 2,
  heightPreference: 3,
  size: 4,
} as const

export const sourceSize = globalField.size + slotCount * slotField.size

export function slotFieldIndex(slot: number, field: number): number {
  invariant(Number.isInteger(slot) && slot >= 0 && slot < slotCount, 'slot index is invalid')
  invariant(Number.isInteger(field) && field >= 0 && field < slotField.size, 'slot field is invalid')
  return globalField.size + slot * slotField.size + field
}

// One entry per dimension of the generative field vector: where it lives in the
// source table and the closed range it may take. Search samples inside these
// ranges, interpolation stays inside them by convexity, and gradient steps
// clamp back into them.
export type FieldDescriptor = {
  name: string
  source: number
  minimum: number
  maximum: number
}

function describeGenerativeFields(): FieldDescriptor[] {
  const descriptors: FieldDescriptor[] = [
    {name: 'paddingX', source: globalField.paddingX, minimum: 0, maximum: 120},
    {name: 'paddingY', source: globalField.paddingY, minimum: 0, maximum: 120},
    {name: 'gapX', source: globalField.gapX, minimum: 0, maximum: 0.06},
    {name: 'gapY', source: globalField.gapY, minimum: 0, maximum: 60},
    {name: 'baseSpan', source: globalField.baseSpan, minimum: 0.03, maximum: 1},
    {name: 'spanFromImportance', source: globalField.spanFromImportance, minimum: 0, maximum: 0.9},
    {name: 'baseHeight', source: globalField.baseHeight, minimum: 24, maximum: 720},
    {name: 'heightFromImportance', source: globalField.heightFromImportance, minimum: 0, maximum: 700},
  ]
  const slotFieldNames = ['importance', 'orderKey', 'visibility', 'heightPreference'] as const
  for (let slot = 0; slot < slotCount; slot++) {
    for (let field = 0; field < slotField.size; field++) {
      descriptors.push({
        name: `slot${slot}.${slotFieldNames[field]!}`,
        source: slotFieldIndex(slot, field),
        minimum: 0,
        maximum: 1,
      })
    }
  }
  return descriptors
}

export const generativeFields: readonly FieldDescriptor[] = describeGenerativeFields()

export const generativeFieldCount = generativeFields.length

export type LayoutContext = {
  viewportWidth: number
  viewportHeight: number
  wrapSlack: number
  minVisibleWidth: number
  minVisibleHeight: number
}

export const defaultContext: LayoutContext = {
  viewportWidth: 1280,
  viewportHeight: 800,
  wrapSlack: 0.004,
  minVisibleWidth: 48,
  minVisibleHeight: 32,
}

// Source table ranges the circuit assumes. They cover the generative ranges
// above plus the context values; evaluate() validates every source against
// them, so samplers and optimizers must clamp before evaluating.
export const sourceRanges: readonly (readonly [minimum: number, maximum: number])[] = (() => {
  const ranges: [number, number][] = Array.from({length: sourceSize}, () => [0, 1])
  ranges[globalField.viewportWidth] = [1, 8192]
  ranges[globalField.viewportHeight] = [1, 8192]
  ranges[globalField.paddingX] = [0, 200]
  ranges[globalField.paddingY] = [0, 200]
  ranges[globalField.gapX] = [0, 0.1]
  ranges[globalField.gapY] = [0, 100]
  ranges[globalField.baseSpan] = [0.02, 1]
  ranges[globalField.spanFromImportance] = [0, 1]
  ranges[globalField.baseHeight] = [1, 1024]
  ranges[globalField.heightFromImportance] = [0, 1024]
  ranges[globalField.wrapSlack] = [0.0005, 0.05]
  ranges[globalField.minVisibleWidth] = [0, 512]
  ranges[globalField.minVisibleHeight] = [0, 512]
  ranges[globalField.unit] = [1, 1]
  return ranges
})()

export function buildSource(fields: Float64Array, context: LayoutContext): Float32Array {
  invariant(fields.length === generativeFieldCount, 'generative field vector has the wrong length')
  const source = new Float32Array(sourceSize)
  source[globalField.viewportWidth] = context.viewportWidth
  source[globalField.viewportHeight] = context.viewportHeight
  source[globalField.wrapSlack] = context.wrapSlack
  source[globalField.minVisibleWidth] = context.minVisibleWidth
  source[globalField.minVisibleHeight] = context.minVisibleHeight
  source[globalField.unit] = 1
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    const descriptor = generativeFields[dimension]!
    const value = fields[dimension]!
    invariant(
      value >= descriptor.minimum && value <= descriptor.maximum,
      `field ${descriptor.name} is outside its declared range`,
    )
    source[descriptor.source] = value
  }
  return source
}

export function clampFields(fields: Float64Array) {
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    const descriptor = generativeFields[dimension]!
    fields[dimension] = Math.min(
      descriptor.maximum,
      Math.max(descriptor.minimum, fields[dimension]!),
    )
  }
}

export function interpolateFields(a: Float64Array, b: Float64Array, t: number): Float64Array {
  invariant(a.length === generativeFieldCount && b.length === generativeFieldCount, 'field vectors have the wrong length')
  const result = new Float64Array(generativeFieldCount)
  for (let dimension = 0; dimension < generativeFieldCount; dimension++) {
    result[dimension] = a[dimension]! + (b[dimension]! - a[dimension]!) * t
  }
  return result
}

// Convenience constructor for readable authored field vectors.
export type AuthoredFields = {
  paddingX: number
  paddingY: number
  gapX: number
  gapY: number
  baseSpan: number
  spanFromImportance: number
  baseHeight: number
  heightFromImportance: number
  slots: {
    importance: number
    orderKey: number
    visibility: number
    heightPreference: number
  }[]
}

export function makeFieldVector(authored: AuthoredFields): Float64Array {
  invariant(authored.slots.length === slotCount, 'authored slots have the wrong length')
  const fields = new Float64Array(generativeFieldCount)
  const globalValues = [
    authored.paddingX,
    authored.paddingY,
    authored.gapX,
    authored.gapY,
    authored.baseSpan,
    authored.spanFromImportance,
    authored.baseHeight,
    authored.heightFromImportance,
  ]
  for (let dimension = 0; dimension < globalValues.length; dimension++) {
    fields[dimension] = globalValues[dimension]!
  }
  for (let slot = 0; slot < slotCount; slot++) {
    const base = globalValues.length + slot * slotField.size
    const slotValues = authored.slots[slot]!
    fields[base + slotField.importance] = slotValues.importance
    fields[base + slotField.orderKey] = slotValues.orderKey
    fields[base + slotField.visibility] = slotValues.visibility
    fields[base + slotField.heightPreference] = slotValues.heightPreference
  }
  return fields
}

export function fieldDimension(name: string): number {
  const dimension = generativeFields.findIndex(descriptor => descriptor.name === name)
  invariant(dimension >= 0, `unknown generative field ${name}`)
  return dimension
}

function uniformSlots(slots: {
  importance?: number
  visibility?: number
  heightPreference?: number
}[]): AuthoredFields['slots'] {
  return Array.from({length: slotCount}, (_, slot) => ({
    importance: slots[slot]?.importance ?? 0.2,
    orderKey: (slot + 0.5) / slotCount,
    visibility: slots[slot]?.visibility ?? 1,
    heightPreference: slots[slot]?.heightPreference ?? 0.5,
  }))
}

// Five authored layout families, all at the default 1280x800 context, all
// produced by the same compiled circuit with zero recompilation. The numbers
// were tuned against the oracle so that every family is valid.
export function authoredFamilies(): {name: string, fields: Float64Array}[] {
  const grid = makeFieldVector({
    paddingX: 24, paddingY: 24, gapX: 0.015, gapY: 14,
    baseSpan: 0.235, spanFromImportance: 0, baseHeight: 177, heightFromImportance: 0,
    slots: uniformSlots(Array.from({length: slotCount}, () => ({}))),
  })

  const heroRail = makeFieldVector({
    paddingX: 24, paddingY: 24, gapX: 0.015, gapY: 12,
    baseSpan: 0.2, spanFromImportance: 0.58, baseHeight: 70, heightFromImportance: 120,
    slots: uniformSlots(Array.from({length: slotCount}, (_, slot) => ({
      importance: slot === 0 ? 0.95 : 0.12,
    }))),
  })

  const mosaicImportance = [
    0.9, 0.35, 0.65, 0.2, 0.5, 0.8, 0.15, 0.4,
    0.7, 0.25, 0.55, 0.3, 0.85, 0.45, 0.6, 0.1,
  ]
  const mosaicPreference = [
    0.8, 0.4, 0.6, 0.3, 0.5, 0.7, 0.3, 0.5,
    0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6, 0.3,
  ]
  const mosaic = makeFieldVector({
    paddingX: 20, paddingY: 20, gapX: 0.012, gapY: 12,
    baseSpan: 0.18, spanFromImportance: 0.25, baseHeight: 48, heightFromImportance: 56,
    slots: uniformSlots(Array.from({length: slotCount}, (_, slot) => ({
      importance: mosaicImportance[slot]!,
      heightPreference: mosaicPreference[slot]!,
    }))),
  })

  const list = makeFieldVector({
    paddingX: 40, paddingY: 24, gapX: 0.01, gapY: 12,
    baseSpan: 1, spanFromImportance: 0, baseHeight: 110, heightFromImportance: 0,
    slots: uniformSlots(Array.from({length: slotCount}, (_, slot) => ({
      importance: 0.3,
      visibility: slot < 6 ? 1 : 0,
    }))),
  })

  const filmstrip = makeFieldVector({
    paddingX: 24, paddingY: 60, gapX: 0.006, gapY: 10,
    baseSpan: 0.0555, spanFromImportance: 0, baseHeight: 640, heightFromImportance: 0,
    slots: uniformSlots(Array.from({length: slotCount}, () => ({}))),
  })

  return [
    {name: 'grid', fields: grid},
    {name: 'heroRail', fields: heroRail},
    {name: 'mosaic', fields: mosaic},
    {name: 'list', fields: list},
    {name: 'filmstrip', fields: filmstrip},
  ]
}

// The hero migration pair: identical fields except that the importance peak
// sits on slot 2 in A and slot 11 in B. The straight line between them moves
// the importance mass; halfway both slots hold medium importance with exactly
// equal areas. Measured at t = 0.5: the two co-heroes sit in separate rows
// (slot 2 next to slot 3 near the top, slot 11 next to slot 10 further down),
// each sharing its row with one small slot.
export function heroMigrationPair(): {a: Float64Array, b: Float64Array} {
  function withHero(heroSlot: number) {
    return makeFieldVector({
      paddingX: 24, paddingY: 24, gapX: 0.015, gapY: 12,
      baseSpan: 0.2, spanFromImportance: 0.5, baseHeight: 64, heightFromImportance: 96,
      slots: uniformSlots(Array.from({length: slotCount}, (_, slot) => ({
        importance: slot === heroSlot ? 0.95 : 0.12,
      }))),
    })
  }
  return {a: withHero(2), b: withHero(11)}
}
