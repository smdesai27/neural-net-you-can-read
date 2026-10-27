// Duplo-style layout fitness: a handful of weighted heuristic terms over the
// circuit's outputs, in the spirit of Flipboard's layout engine (text density,
// frame fill, pacing heuristics with branch-and-bound selection).
//
// Split of responsibilities, and why: the circuit computes the division-free
// numerators (covered area, content bottom, all violation depths) as wires;
// this file owns every division (fill ratios, means, the Pearson correlation),
// following the repo rule that the scorer/parameter side owns division. Each
// term also has a hand-derived gradient with respect to the slot geometry and
// the two numerator wires, so gradient ascent can seed the circuit's backward
// pass exactly the way src/learned-gallery/trainable-geometry.ts seeds its coordinate losses.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'
import {
  type FlowLayoutOutput,
  geometryField,
} from './flow-layout.ts'
import {globalField, slotCount, slotField, slotFieldIndex} from './fields.ts'

// A slot smaller than this area (in square pixels) reads as cramped; the
// comfort term saturates at this size.
const comfortArea = 140 * 100

// A layout showing at least this many slots earns the full content richness
// score. Without this term, gradient ascent collapses layouts into two or
// three viewport-filling slabs that max out every other term (an instructive
// reward-hacking episode, documented in the README).
const richSlotCount = 10

export const fitnessWeights = {
  fill: 0.25,
  verticalFit: 0.2,
  importanceMatch: 0.15,
  balanceX: 0.1,
  sizeComfort: 0.1,
  rowEvenness: 0.05,
  contentRichness: 0.15,
} as const

// Alternative weightings plug in here; the reward-hacking characterization
// test re-runs gradient ascent with the pre-richness weighting to reproduce
// the slab collapse.
export type FitnessWeights = Record<keyof typeof fitnessWeights, number>

export type LayoutSnapshot = {
  x: Float64Array
  y: Float64Array
  width: Float64Array
  height: Float64Array
  visible: boolean[]
  importance: Float64Array
  viewportWidth: number
  viewportHeight: number
  paddingX: number
  paddingY: number
  innerWidth: number
  innerHeight: number
  coveredArea: number
  contentBottom: number
  valid: boolean
  violationCount: number
  overlapViolationCount: number
  containViolationCount: number
  sizeViolationCount: number
  violationDepth: number
}

export function extractSnapshot(
  values: Float32Array,
  output: FlowLayoutOutput,
  source: Float32Array,
): LayoutSnapshot {
  const x = new Float64Array(slotCount)
  const y = new Float64Array(slotCount)
  const width = new Float64Array(slotCount)
  const height = new Float64Array(slotCount)
  const visible: boolean[] = []
  const importance = new Float64Array(slotCount)
  for (let slot = 0; slot < slotCount; slot++) {
    const start = slot * geometryField.size
    x[slot] = values[output.slotGeometry[start + geometryField.x]!]!
    y[slot] = values[output.slotGeometry[start + geometryField.y]!]!
    width[slot] = values[output.slotGeometry[start + geometryField.width]!]!
    height[slot] = values[output.slotGeometry[start + geometryField.height]!]!
    visible.push(values[output.visibleBits[slot]!]! === 1)
    importance[slot] = source[slotFieldIndex(slot, slotField.importance)]!
  }
  const viewportWidth = source[globalField.viewportWidth]!
  const viewportHeight = source[globalField.viewportHeight]!
  const paddingX = source[globalField.paddingX]!
  const paddingY = source[globalField.paddingY]!
  return {
    x, y, width, height, visible, importance,
    viewportWidth, viewportHeight, paddingX, paddingY,
    innerWidth: Math.max(1, viewportWidth - 2 * paddingX),
    innerHeight: Math.max(1, viewportHeight - 2 * paddingY),
    coveredArea: values[output.coveredArea]!,
    contentBottom: values[output.contentBottom]!,
    valid: values[output.validBit]! === 1,
    violationCount: values[output.violationCount]!,
    overlapViolationCount: values[output.overlapViolationCount]!,
    containViolationCount: values[output.containViolationCount]!,
    sizeViolationCount: values[output.sizeViolationCount]!,
    violationDepth: values[output.violationDepth]!,
  }
}

export type FitnessGradient = {
  x: Float64Array
  y: Float64Array
  width: Float64Array
  height: Float64Array
  // Straight-through seed for the visibility bits (content richness wants
  // more slots shown); meaningful only through the STE backward pass.
  visible: Float64Array
  importance: Float64Array
  coveredArea: number
  contentBottom: number
  innerWidth: number
  innerHeight: number
  viewportWidth: number
  viewportHeight: number
  paddingY: number
}

export type FitnessResult = {
  value: number
  terms: Record<keyof typeof fitnessWeights, number>
  gradient: FitnessGradient
}

// Group visible slots into rows by their y coordinate (values within one pixel
// share a row; flow rows are separated by at least the row height).
function groupRows(snapshot: LayoutSnapshot): number[][] {
  const orderedSlots = Array.from({length: slotCount}, (_, slot) => slot)
    .filter(slot => snapshot.visible[slot]!)
    .sort((a, b) => snapshot.y[a]! - snapshot.y[b]!)
  const rows: number[][] = []
  let currentTop = Number.NEGATIVE_INFINITY
  for (const slot of orderedSlots) {
    if (snapshot.y[slot]! - currentTop > 1) {
      rows.push([])
      currentTop = snapshot.y[slot]!
    }
    rows.at(-1)!.push(slot)
  }
  return rows
}

export function scoreFitness(
  snapshot: LayoutSnapshot,
  weights: FitnessWeights = fitnessWeights,
): FitnessResult {
  const gradient: FitnessGradient = {
    x: new Float64Array(slotCount),
    y: new Float64Array(slotCount),
    width: new Float64Array(slotCount),
    height: new Float64Array(slotCount),
    visible: new Float64Array(slotCount),
    importance: new Float64Array(slotCount),
    coveredArea: 0,
    contentBottom: 0,
    innerWidth: 0,
    innerHeight: 0,
    viewportWidth: 0,
    viewportHeight: 0,
    paddingY: 0,
  }
  const visibleSlots = Array.from({length: slotCount}, (_, slot) => slot)
    .filter(slot => snapshot.visible[slot]!)
  const visibleCount = visibleSlots.length
  const terms = {
    fill: 0,
    verticalFit: 0,
    importanceMatch: 0,
    balanceX: 0,
    sizeComfort: 0,
    rowEvenness: 0,
    contentRichness: 0,
  }
  if (visibleCount === 0) {
    // Even an empty layout wants more content; seed the visibility bits.
    for (let slot = 0; slot < slotCount; slot++) {
      gradient.visible[slot] = weights.contentRichness / richSlotCount
    }
    return {value: 0, terms, gradient}
  }

  // Content richness: reward showing a healthy number of slots.
  terms.contentRichness = Math.min(1, visibleCount / richSlotCount)
  if (visibleCount < richSlotCount) {
    for (let slot = 0; slot < slotCount; slot++) {
      gradient.visible[slot] = weights.contentRichness / richSlotCount
    }
  }

  // Fill: covered area over the inner box area, saturating at 1.
  const usableArea = snapshot.innerWidth * snapshot.innerHeight
  const fillRatio = snapshot.coveredArea / usableArea
  terms.fill = Math.min(1, fillRatio)
  if (fillRatio < 1) {
    gradient.coveredArea += weights.fill / usableArea
    gradient.innerWidth -= weights.fill * fillRatio / snapshot.innerWidth
    gradient.innerHeight -= weights.fill * fillRatio / snapshot.innerHeight
  }

  // Vertical fit: the content bottom should land at the padded bottom edge;
  // both unused space and overflow lose score.
  const bottomLimit = snapshot.viewportHeight - snapshot.paddingY
  const bottomError = snapshot.contentBottom - bottomLimit
  const verticalMiss = Math.abs(bottomError) / snapshot.innerHeight
  terms.verticalFit = Math.max(0, 1 - verticalMiss)
  if (terms.verticalFit > 0 && bottomError !== 0) {
    gradient.contentBottom +=
      weights.verticalFit * (-Math.sign(bottomError) / snapshot.innerHeight)
    const limitGradient = weights.verticalFit * Math.sign(bottomError) / snapshot.innerHeight
    gradient.viewportHeight += limitGradient
    gradient.paddingY -= limitGradient
    gradient.innerHeight += weights.verticalFit * verticalMiss / snapshot.innerHeight
  }

  // Horizontal balance: the area-weighted centroid should sit near the middle.
  const areas = visibleSlots.map(slot => snapshot.width[slot]! * snapshot.height[slot]!)
  const totalArea = areas.reduce((sum, area) => sum + area, 0)
  if (totalArea > 1e-9) {
    const centers = visibleSlots.map(slot => snapshot.x[slot]! + snapshot.width[slot]! / 2)
    let weightedCenter = 0
    for (let index = 0; index < visibleSlots.length; index++) {
      weightedCenter += areas[index]! * centers[index]!
    }
    const centroid = weightedCenter / totalArea
    const middle = snapshot.viewportWidth / 2
    const halfInner = snapshot.innerWidth / 2
    const offset = (centroid - middle) / halfInner
    terms.balanceX = Math.max(0, 1 - Math.abs(offset))
    if (terms.balanceX > 0 && offset !== 0) {
      const scale = weights.balanceX * (-Math.sign(offset) / halfInner)
      gradient.viewportWidth -= scale / 2
      gradient.innerWidth += weights.balanceX * Math.abs(offset) / snapshot.innerWidth
      for (let index = 0; index < visibleSlots.length; index++) {
        const slot = visibleSlots[index]!
        const centerPull = areas[index]! / totalArea
        const areaPull = (centers[index]! - centroid) / totalArea
        gradient.x[slot] = gradient.x[slot]! + scale * centerPull
        gradient.width[slot] = gradient.width[slot]!
          + scale * (centerPull * 0.5 + areaPull * snapshot.height[slot]!)
        gradient.height[slot] = gradient.height[slot]!
          + scale * areaPull * snapshot.width[slot]!
      }
    }
  }

  // Importance match: important slots should hold more area. Pearson
  // correlation between importance and area over visible slots, mapped to
  // [0, 1]. Neutral 0.5 when there is nothing to correlate.
  terms.importanceMatch = 0.5
  if (visibleCount >= 2) {
    const meanImportance = visibleSlots
      .reduce((sum, slot) => sum + snapshot.importance[slot]!, 0) / visibleCount
    const meanArea = totalArea / visibleCount
    let importanceSpread = 0
    let areaSpread = 0
    let crossSpread = 0
    for (let index = 0; index < visibleSlots.length; index++) {
      const slot = visibleSlots[index]!
      const centeredImportance = snapshot.importance[slot]! - meanImportance
      const centeredArea = areas[index]! - meanArea
      importanceSpread += centeredImportance * centeredImportance
      areaSpread += centeredArea * centeredArea
      crossSpread += centeredImportance * centeredArea
    }
    if (importanceSpread > 1e-9 && areaSpread > 1e-3) {
      const normalizer = Math.sqrt(importanceSpread * areaSpread)
      const correlation = crossSpread / normalizer
      terms.importanceMatch = (correlation + 1) / 2
      const scale = weights.importanceMatch * 0.5
      for (let index = 0; index < visibleSlots.length; index++) {
        const slot = visibleSlots[index]!
        const centeredImportance = snapshot.importance[slot]! - meanImportance
        const centeredArea = areas[index]! - meanArea
        const correlationByArea =
          centeredImportance / normalizer - correlation * centeredArea / areaSpread
        gradient.importance[slot] = scale * (
          centeredArea / normalizer - correlation * centeredImportance / importanceSpread
        )
        gradient.width[slot] = gradient.width[slot]!
          + scale * correlationByArea * snapshot.height[slot]!
        gradient.height[slot] = gradient.height[slot]!
          + scale * correlationByArea * snapshot.width[slot]!
      }
    }
  }

  // Size comfort: every visible slot should reach a comfortable area.
  for (let index = 0; index < visibleSlots.length; index++) {
    const slot = visibleSlots[index]!
    const ratio = areas[index]! / comfortArea
    terms.sizeComfort += Math.min(1, ratio) / visibleCount
    if (ratio < 1) {
      const scale = weights.sizeComfort / (visibleCount * comfortArea)
      gradient.width[slot] = gradient.width[slot]! + scale * snapshot.height[slot]!
      gradient.height[slot] = gradient.height[slot]! + scale * snapshot.width[slot]!
    }
  }

  // Row evenness: the ragged right edges across rows should align.
  const rows = groupRows(snapshot)
  if (rows.length <= 1) {
    terms.rowEvenness = 1
  } else {
    let widestRowEdge = Number.NEGATIVE_INFINITY
    let narrowestRowEdge = Number.POSITIVE_INFINITY
    let widestSlot = -1
    let narrowestSlot = -1
    for (const row of rows) {
      let rowEdge = Number.NEGATIVE_INFINITY
      let rowEdgeSlot = -1
      for (const slot of row) {
        const edge = snapshot.x[slot]! + snapshot.width[slot]!
        if (edge > rowEdge) {
          rowEdge = edge
          rowEdgeSlot = slot
        }
      }
      if (rowEdge > widestRowEdge) {
        widestRowEdge = rowEdge
        widestSlot = rowEdgeSlot
      }
      if (rowEdge < narrowestRowEdge) {
        narrowestRowEdge = rowEdge
        narrowestSlot = rowEdgeSlot
      }
    }
    const spread = (widestRowEdge - narrowestRowEdge) / snapshot.innerWidth
    terms.rowEvenness = Math.max(0, 1 - spread)
    if (terms.rowEvenness > 0 && spread > 0 && widestSlot !== narrowestSlot) {
      const scale = weights.rowEvenness / snapshot.innerWidth
      gradient.innerWidth += weights.rowEvenness * spread / snapshot.innerWidth
      gradient.x[widestSlot] = gradient.x[widestSlot]! - scale
      gradient.width[widestSlot] = gradient.width[widestSlot]! - scale
      gradient.x[narrowestSlot] = gradient.x[narrowestSlot]! + scale
      gradient.width[narrowestSlot] = gradient.width[narrowestSlot]! + scale
    }
  }

  const value =
    weights.fill * terms.fill
    + weights.verticalFit * terms.verticalFit
    + weights.importanceMatch * terms.importanceMatch
    + weights.balanceX * terms.balanceX
    + weights.sizeComfort * terms.sizeComfort
    + weights.rowEvenness * terms.rowEvenness
    + weights.contentRichness * terms.contentRichness
  invariant(Number.isFinite(value), 'fitness must be finite')
  return {value, terms, gradient}
}

// Ranking score for search: valid layouts rank by fitness in [0, 1]; invalid
// layouts rank strictly below every valid one, less negative when closer to
// validity, so a search can still say which invalid candidate is nearest.
export function selectionScore(snapshot: LayoutSnapshot, fitnessValue: number): number {
  if (snapshot.valid) return fitnessValue
  return -1 - Math.min(1, snapshot.violationDepth / 1000)
}
