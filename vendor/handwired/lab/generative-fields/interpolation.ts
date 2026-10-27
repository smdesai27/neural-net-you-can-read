// The interpolation study: walk straight lines between pairs of valid field
// vectors, measure how often the intermediate layouts stay valid, and repair
// the invalid intermediates by projecting back to validity with the circuit's
// own violation outputs.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'
import {buildSource, interpolateFields, type LayoutContext} from './fields.ts'
import type {FlowLayoutMachine} from './flow-layout.ts'
import {projectToValidity, type ProjectionOptions} from './optimize.ts'

export type PathPoint = {
  t: number
  validBefore: boolean
  violationCountBefore: number
  // Per-category violation counts before projection, straight from the
  // circuit's category wires.
  overlapCountBefore: number
  containCountBefore: number
  sizeCountBefore: number
  violationDepthBefore: number
  validAfter: boolean
  projectionSteps: number
  movedDistance: number
}

export type PathResult = {
  points: PathPoint[]
  validBeforeCount: number
  validAfterCount: number
}

export function walkPath(
  machine: FlowLayoutMachine,
  context: LayoutContext,
  fieldsA: Float64Array,
  fieldsB: Float64Array,
  sampleCount: number,
  projection: ProjectionOptions = {},
): PathResult {
  invariant(sampleCount >= 2, 'a path needs at least its two endpoints')
  const points: PathPoint[] = []
  let validBeforeCount = 0
  let validAfterCount = 0
  for (let sample = 0; sample < sampleCount; sample++) {
    const t = sample / (sampleCount - 1)
    const fields = interpolateFields(fieldsA, fieldsB, t)
    const source = buildSource(fields, context)
    const values = machine.evaluate(source)
    const validBefore = values[machine.output.validBit]! === 1
    const violationCountBefore = values[machine.output.violationCount]!
    const overlapCountBefore = values[machine.output.overlapViolationCount]!
    const containCountBefore = values[machine.output.containViolationCount]!
    const sizeCountBefore = values[machine.output.sizeViolationCount]!
    const violationDepthBefore = values[machine.output.violationDepth]!
    let validAfter = validBefore
    let projectionSteps = 0
    let movedDistance = 0
    if (!validBefore) {
      const projected = projectToValidity(machine, context, fields, projection)
      validAfter = projected.valid
      projectionSteps = projected.steps
      movedDistance = projected.movedDistance
    }
    if (validBefore) validBeforeCount++
    if (validAfter) validAfterCount++
    points.push({
      t,
      validBefore,
      violationCountBefore,
      overlapCountBefore,
      containCountBefore,
      sizeCountBefore,
      violationDepthBefore,
      validAfter,
      projectionSteps,
      movedDistance,
    })
  }
  return {points, validBeforeCount, validAfterCount}
}

export type StudyResult = {
  pairCount: number
  pointsPerPath: number
  totalPoints: number
  validBeforeCount: number
  validAfterCount: number
  validBeforeFraction: number
  validAfterFraction: number
  // Paths that were already valid at every sampled point.
  fullyValidPathCount: number
  // Points with at least one violation before projection, split by category.
  // A point can count toward more than one category.
  overlapPointCount: number
  containPointCount: number
  sizePointCount: number
  // Distribution of per-path valid-before fractions, sorted ascending.
  pathValidBeforeFractions: number[]
  // Projection statistics over the points that needed repair.
  repairedCount: number
  failedRepairCount: number
  meanProjectionSteps: number
  meanMovedDistance: number
  maxMovedDistance: number
}

export function runInterpolationStudy(
  machine: FlowLayoutMachine,
  context: LayoutContext,
  pairs: readonly {a: Float64Array, b: Float64Array}[],
  options: {samplesPerPath: number, projection?: ProjectionOptions},
): StudyResult {
  const {samplesPerPath} = options
  let validBeforeCount = 0
  let validAfterCount = 0
  let fullyValidPathCount = 0
  let overlapPointCount = 0
  let containPointCount = 0
  let sizePointCount = 0
  let repairedCount = 0
  let failedRepairCount = 0
  let projectionStepSum = 0
  let movedDistanceSum = 0
  let maxMovedDistance = 0
  const pathValidBeforeFractions: number[] = []
  for (const pair of pairs) {
    const path = walkPath(
      machine,
      context,
      pair.a,
      pair.b,
      samplesPerPath,
      options.projection ?? {},
    )
    validBeforeCount += path.validBeforeCount
    validAfterCount += path.validAfterCount
    if (path.validBeforeCount === samplesPerPath) fullyValidPathCount++
    pathValidBeforeFractions.push(path.validBeforeCount / samplesPerPath)
    for (const point of path.points) {
      if (point.overlapCountBefore > 0) overlapPointCount++
      if (point.containCountBefore > 0) containPointCount++
      if (point.sizeCountBefore > 0) sizePointCount++
      if (point.validBefore) continue
      if (point.validAfter) {
        repairedCount++
        projectionStepSum += point.projectionSteps
        movedDistanceSum += point.movedDistance
        maxMovedDistance = Math.max(maxMovedDistance, point.movedDistance)
      } else {
        failedRepairCount++
      }
    }
  }
  pathValidBeforeFractions.sort((a, b) => a - b)
  const totalPoints = pairs.length * samplesPerPath
  return {
    pairCount: pairs.length,
    pointsPerPath: samplesPerPath,
    totalPoints,
    validBeforeCount,
    validAfterCount,
    validBeforeFraction: validBeforeCount / totalPoints,
    validAfterFraction: validAfterCount / totalPoints,
    fullyValidPathCount,
    overlapPointCount,
    containPointCount,
    sizePointCount,
    pathValidBeforeFractions,
    repairedCount,
    failedRepairCount,
    meanProjectionSteps: repairedCount === 0 ? 0 : projectionStepSum / repairedCount,
    meanMovedDistance: repairedCount === 0 ? 0 : movedDistanceSum / repairedCount,
    maxMovedDistance,
  }
}
