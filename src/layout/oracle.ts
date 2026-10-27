// Plain float64 reference implementation of the flow layout, its validity
// checks, and its fitness numerators. The circuit in flow-layout.ts must agree
// with this oracle: geometry within an FP32 accumulation tolerance, decision
// bits exactly whenever the decision margins are clear of the FP32 rounding
// band. The oracle reports the smallest margin it saw per decision category so
// tests can skip knife-edge samples instead of asserting on them.

// Fork of vendor/handwired/lab/generative-fields/oracle.ts with the same
// minimum-tile-width change as circuit.ts, in float64.
import {invariant} from '../../vendor/handwired/src/runtime/numeric-ui-core.ts'
import {extendedSourceSize, globalField, invInnerWidthCell, minTilePxCell, slotCount, slotField, slotFieldIndex} from './schema.ts'
import {violationSlackPixels} from './circuit.ts'

export type OracleSlot = {
  x: number
  y: number
  width: number
  height: number
  visible: boolean
  rank: number
  span: number
}

export type OracleResult = {
  slots: OracleSlot[]
  overlapViolationCount: number
  containViolationCount: number
  sizeViolationCount: number
  violationCount: number
  valid: boolean
  violationDepth: number
  coveredArea: number
  contentBottom: number
  // Smallest |margin| per decision category, for knife-edge detection.
  margins: {
    orderKey: number
    visibility: number
    wrap: number
    violation: number
  }
}

export function evaluateOracle(source: Float32Array): OracleResult {
  invariant(source.length === extendedSourceSize, 'oracle source table has the wrong length')
  const viewportWidth = source[globalField.viewportWidth]!
  const viewportHeight = source[globalField.viewportHeight]!
  const paddingX = source[globalField.paddingX]!
  const paddingY = source[globalField.paddingY]!
  const gapX = source[globalField.gapX]!
  const gapY = source[globalField.gapY]!
  const baseSpan = source[globalField.baseSpan]!
  const spanFromImportance = source[globalField.spanFromImportance]!
  const baseHeight = source[globalField.baseHeight]!
  const heightFromImportance = source[globalField.heightFromImportance]!
  const wrapSlack = source[globalField.wrapSlack]!
  const minVisibleWidth = source[globalField.minVisibleWidth]!
  const minVisibleHeight = source[globalField.minVisibleHeight]!

  const margins = {
    orderKey: Number.POSITIVE_INFINITY,
    visibility: Number.POSITIVE_INFINITY,
    wrap: Number.POSITIVE_INFINITY,
    violation: Number.POSITIVE_INFINITY,
  }

  const importance = Array.from({length: slotCount}, (_, slot) =>
    source[slotFieldIndex(slot, slotField.importance)]!)
  const orderKey = Array.from({length: slotCount}, (_, slot) =>
    source[slotFieldIndex(slot, slotField.orderKey)]!)
  const visible = Array.from({length: slotCount}, (_, slot) => {
    const visibility = source[slotFieldIndex(slot, slotField.visibility)]!
    margins.visibility = Math.min(margins.visibility, Math.abs(visibility - 0.5))
    return visibility >= 0.5
  })
  const heightPreference = Array.from({length: slotCount}, (_, slot) =>
    source[slotFieldIndex(slot, slotField.heightPreference)]!)

  const minSpan = source[minTilePxCell]! * source[invInnerWidthCell]!
  const spans = importance.map(value => Math.min(1, Math.max(baseSpan + spanFromImportance * value, minSpan)))
  const heights = importance.map((value, slot) =>
    (baseHeight + heightFromImportance * value) * (0.5 + heightPreference[slot]!))

  // Rank = number of visible slots ordered before this slot; the order is
  // ascending (order key, slot index). Track the smallest key distance among
  // pairs where at least one side is visible: those are the comparisons whose
  // knife edges can change a visible slot's rank.
  const ranks = Array.from({length: slotCount}, (_, slot) => {
    let rank = 0
    for (let other = 0; other < slotCount; other++) {
      if (other === slot || !visible[other]!) continue
      const otherFirst = other < slot
        ? orderKey[other]! <= orderKey[slot]!
        : orderKey[other]! < orderKey[slot]!
      if (otherFirst) rank++
    }
    return rank
  })
  for (let earlier = 0; earlier < slotCount; earlier++) {
    for (let later = earlier + 1; later < slotCount; later++) {
      if (!visible[earlier]! && !visible[later]!) continue
      margins.orderKey = Math.min(
        margins.orderKey,
        Math.abs(orderKey[earlier]! - orderKey[later]!),
      )
    }
  }

  // Greedy line breaking over rank positions, in fraction units horizontally
  // and pixels vertically, mirroring the circuit's sequential flow.
  const slotAtPosition = Array<number>(slotCount).fill(-1)
  for (let slot = 0; slot < slotCount; slot++) {
    if (visible[slot]!) slotAtPosition[ranks[slot]!] = slot
  }
  const slots: OracleSlot[] = Array.from({length: slotCount}, (_, slot) => ({
    x: 0, y: 0, width: 0, height: 0, visible: visible[slot]!, rank: ranks[slot]!, span: spans[slot]!,
  }))
  const innerWidth = Math.max(0, viewportWidth - 2 * paddingX)
  let cursor = 0
  let rowTop = paddingY
  let rowBottom = paddingY
  for (let position = 0; position < slotCount; position++) {
    const slot = slotAtPosition[position]!
    if (slot < 0) break
    const span = spans[slot]!
    const height = heights[slot]!
    margins.wrap = Math.min(margins.wrap, Math.abs(cursor + span - 1 - wrapSlack))
    const wrap = cursor + span >= 1 + wrapSlack
    const x = wrap ? 0 : cursor
    const top = wrap ? rowBottom + gapY : rowTop
    const bottomCandidate = top + height
    const bottom = wrap ? bottomCandidate : Math.max(rowBottom, bottomCandidate)
    slots[slot]!.x = paddingX + innerWidth * x
    slots[slot]!.y = top
    slots[slot]!.width = innerWidth * span
    slots[slot]!.height = height
    cursor = x + span + gapX
    rowTop = top
    rowBottom = bottom
  }

  // Validity, mirroring the circuit's slack thresholds.
  let overlapViolationCount = 0
  let containViolationCount = 0
  let sizeViolationCount = 0
  let violationDepth = 0
  function checkViolation(depth: number): boolean {
    margins.violation = Math.min(margins.violation, Math.abs(depth - violationSlackPixels))
    violationDepth += depth
    return depth >= violationSlackPixels
  }
  for (let earlier = 0; earlier < slotCount; earlier++) {
    for (let later = earlier + 1; later < slotCount; later++) {
      const a = slots[earlier]!
      const b = slots[later]!
      const penetrationX = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)
      const penetrationY = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y)
      const bothVisible = a.visible && b.visible
      const depth = bothVisible ? Math.max(0, Math.min(penetrationX, penetrationY)) : 0
      if (checkViolation(depth)) overlapViolationCount++
    }
  }
  const bottomLimit = viewportHeight - paddingY
  const rightLimit = paddingX + innerWidth + innerWidth * wrapSlack
  for (let slot = 0; slot < slotCount; slot++) {
    const item = slots[slot]!
    const bottomOvershoot = item.visible ? Math.max(0, item.y + item.height - bottomLimit) : 0
    if (checkViolation(bottomOvershoot)) containViolationCount++
    const rightOvershoot = item.visible ? Math.max(0, item.x + item.width - rightLimit) : 0
    if (checkViolation(rightOvershoot)) containViolationCount++
  }
  for (let slot = 0; slot < slotCount; slot++) {
    const item = slots[slot]!
    const widthDeficit = item.visible
      ? Math.max(0, minVisibleWidth - innerWidth * item.span)
      : 0
    if (checkViolation(widthDeficit)) sizeViolationCount++
    const heightDeficit = item.visible ? Math.max(0, minVisibleHeight - heights[slot]!) : 0
    if (checkViolation(heightDeficit)) sizeViolationCount++
  }

  let coveredArea = 0
  let contentBottom = 0
  for (const item of slots) {
    if (!item.visible) continue
    coveredArea += item.width * item.height
    contentBottom = Math.max(contentBottom, item.y + item.height)
  }

  const violationCount = overlapViolationCount + containViolationCount + sizeViolationCount
  return {
    slots,
    overlapViolationCount,
    containViolationCount,
    sizeViolationCount,
    violationCount,
    valid: violationCount === 0,
    violationDepth,
    coveredArea,
    contentBottom,
    margins,
  }
}
