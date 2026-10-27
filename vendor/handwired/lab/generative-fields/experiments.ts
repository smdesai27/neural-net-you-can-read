// Runs every experiment and prints the tables the README reports.
// Usage: bun lab/generative-fields/experiments.ts
//
// Everything is seeded; the tests re-run the same configurations and assert
// the same numbers, so the README stays backed by executable checks.

import {
  authoredFamilies,
  buildSource,
  defaultContext,
  fieldDimension,
  generativeFields,
  heroMigrationPair,
  interpolateFields,
  slotCount,
  slotField,
} from './fields.ts'
import {createFlowLayoutMachine, type FlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, type LayoutSnapshot} from './fitness.ts'
import {ascendFitness} from './optimize.ts'
import {createRandom, percentile, runSearch, sampleFieldVector, type SearchResult} from './search.ts'
import {runInterpolationStudy, walkPath} from './interpolation.ts'
import {renderAscii} from './ascii-render.ts'
import {loadPretext, prepareSlotLabels, scoreTextFit, slotLabels} from './text-fitness.ts'

export const searchConfig = {sampleCount: 4000, seed: 0x5EED, keep: 30} as const

export const ascentConfig = {
  startCount: 12,
  startSeed: 0xBEEF,
  steps: 150,
  learningRate: 0.02,
} as const

export const studyConfig = {
  poolSearchTop: 15,
  pairSeed: 0x9A1B,
  randomPairCount: 50,
  samplesPerPath: 101,
  projection: {maxSteps: 40, learningRate: 0.02, proximalWeight: 0.05},
} as const

// The interpolation pool: the five authored families plus the top valid
// searched layouts, deterministically. Pairs are every authored-authored pair
// plus seeded random distinct pairs from the whole pool.
export function buildStudyPairs(search: SearchResult): {
  pool: {name: string, fields: Float64Array}[]
  pairs: {a: Float64Array, b: Float64Array}[]
} {
  const pool = authoredFamilies().map(({name, fields}) => ({name, fields}))
  const authoredCount = pool.length
  let searchedIndex = 0
  for (const candidate of search.top) {
    if (!candidate.snapshot.valid) continue
    if (searchedIndex >= studyConfig.poolSearchTop) break
    pool.push({name: `searched${searchedIndex}`, fields: candidate.fields})
    searchedIndex++
  }
  const pairs: {a: Float64Array, b: Float64Array}[] = []
  for (let first = 0; first < authoredCount; first++) {
    for (let second = first + 1; second < authoredCount; second++) {
      pairs.push({a: pool[first]!.fields, b: pool[second]!.fields})
    }
  }
  const random = createRandom(studyConfig.pairSeed)
  while (pairs.length < authoredCount * (authoredCount - 1) / 2 + studyConfig.randomPairCount) {
    const first = Math.floor(random() * pool.length)
    const second = Math.floor(random() * pool.length)
    if (first === second) continue
    pairs.push({a: pool[first]!.fields, b: pool[second]!.fields})
  }
  return {pool, pairs}
}

function describeFieldVector(fields: Float64Array): string {
  const knobNames = [
    'paddingX', 'paddingY', 'gapX', 'gapY',
    'baseSpan', 'spanFromImportance', 'baseHeight', 'heightFromImportance',
  ]
  const knobs = knobNames
    .map(name => `${name} ${trim(fields[fieldDimension(name)]!)}`)
    .join(', ')
  const rows = ['importance', 'orderKey', 'visibility', 'heightPreference'].map((name, field) => {
    const values = Array.from({length: slotCount}, (_, slot) =>
      trim(fields[knobNames.length + slot * slotField.size + field]!))
    return `  ${name.padEnd(17)} ${values.join(' ')}`
  })
  return `  ${knobs}\n${rows.join('\n')}`
}

function trim(value: number): string {
  const rounded = Math.round(value * 1000) / 1000
  return `${rounded}`
}

function snapshotAt(machine: FlowLayoutMachine, fields: Float64Array): LayoutSnapshot {
  const source = buildSource(fields, defaultContext)
  const values = machine.evaluate(source)
  return extractSnapshot(values, machine.output, source)
}

function runAll() {
  const totalStart = performance.now()
  const buildStart = performance.now()
  const machine = createFlowLayoutMachine()
  const buildMs = performance.now() - buildStart

  // Circuit statistics.
  const opCounts = {affine: 0, reglu: 0, boolean: 0, index: 0}
  for (const operation of machine.circuit.operations) opCounts[operation.kind]++
  let storedWeights = 0
  let maxWidth = 0
  for (const layer of machine.tensorProgram.layers) {
    storedWeights += layer.weights.length
    maxWidth = Math.max(maxWidth, layer.outputSize)
  }
  console.log('## Circuit statistics')
  console.log(`- source cells: ${machine.circuit.sourceSize} (${generativeFields.length} generative fields + context)`)
  console.log(`- operations: ${machine.circuit.operations.length} (${opCounts.affine} affine, ${opCounts.reglu} reglu, ${opCounts.boolean} boolean)`)
  console.log(`- layers: ${machine.tensorProgram.layers.length}, widest layer ${maxWidth} outputs`)
  console.log(`- stored sparse weights: ${storedWeights}`)
  console.log(`- build + compile: ${buildMs.toFixed(0)}ms`)

  // Expressivity: the authored families.
  console.log('\n## Layout families (one compiled circuit, fields only)')
  const operationCountBefore = machine.circuit.operations.length
  const familyRows: string[] = []
  for (const {name, fields} of authoredFamilies()) {
    const snapshot = snapshotAt(machine, fields)
    const fitness = scoreFitness(snapshot)
    const visibleCount = snapshot.visible.filter(Boolean).length
    familyRows.push(
      `| ${name} | ${snapshot.valid ? 'yes' : 'NO'} | ${visibleCount} | ${fitness.value.toFixed(3)} | `
      + `${Object.entries(fitness.terms).map(([term, value]) => `${term} ${value.toFixed(2)}`).join(', ')} |`,
    )
    console.log(`\n### ${name} (valid ${snapshot.valid}, fitness ${fitness.value.toFixed(3)})`)
    console.log(describeFieldVector(fields))
    console.log(renderAscii(snapshot, 56))
  }
  console.log('\n| family | valid | visible | fitness | terms |')
  console.log('| --- | --- | --- | --- | --- |')
  for (const row of familyRows) console.log(row)
  console.log(`- recompilations between families: ${machine.circuit.operations.length - operationCountBefore} new operations (same tensor program object)`)

  // Generation by search.
  console.log('\n## Generation by search')
  const searchStart = performance.now()
  const search = runSearch(machine, defaultContext, searchConfig)
  const searchMs = performance.now() - searchStart
  console.log(`- ${search.sampleCount} samples in ${searchMs.toFixed(0)}ms (seed ${searchConfig.seed})`)
  console.log(`- valid: ${search.validCount} (${(100 * search.validCount / search.sampleCount).toFixed(1)}%)`)
  console.log(`- samples violating each category: overlap ${search.overlapViolationSamples}, containment ${search.containViolationSamples}, minimum size ${search.sizeViolationSamples}`)
  console.log('\n| statistic | fitness among valid |')
  console.log('| --- | --- |')
  for (const [label, fraction] of [['min', 0], ['p25', 0.25], ['median', 0.5], ['p75', 0.75], ['p90', 0.9], ['max', 1]] as const) {
    console.log(`| ${label} | ${percentile(search.validFitness, fraction).toFixed(3)} |`)
  }
  console.log('\ntop 3 searched layouts:')
  for (const candidate of search.top.slice(0, 3)) {
    console.log(`\n### sample ${candidate.sampleIndex} (fitness ${candidate.fitness.toFixed(3)}, visible ${candidate.snapshot.visible.filter(Boolean).length})`)
    console.log(describeFieldVector(candidate.fields))
    console.log(renderAscii(candidate.snapshot, 56))
  }

  // Generation by gradient.
  console.log('\n## Generation by gradient (straight-through estimator)')
  const random = createRandom(ascentConfig.startSeed)
  const ascentStart = performance.now()
  const ascentResults = Array.from({length: ascentConfig.startCount}, () => {
    const startFields = sampleFieldVector(random)
    return ascendFitness(machine, defaultContext, startFields, {
      steps: ascentConfig.steps,
      learningRate: ascentConfig.learningRate,
    })
  })
  const ascentMs = performance.now() - ascentStart
  const ascentScores = ascentResults.map(result => result.score).sort((a, b) => a - b)
  const ascentValidCount = ascentResults.filter(result => result.valid).length
  const ascentEvaluations = ascentResults.reduce((sum, result) => sum + result.evaluations, 0)
  console.log(`- ${ascentConfig.startCount} random starts x ${ascentConfig.steps} Adam steps in ${ascentMs.toFixed(0)}ms (${ascentEvaluations} evaluations)`)
  console.log(`- ended valid: ${ascentValidCount}/${ascentConfig.startCount}`)
  console.log(`- best ${ascentScores.at(-1)!.toFixed(3)}, median ${percentile(ascentScores, 0.5).toFixed(3)}, worst ${ascentScores[0]!.toFixed(3)}`)
  const bestAscent = ascentResults.reduce((best, result) => result.score > best.score ? result : best)
  const bestAscentSnapshot = snapshotAt(machine, bestAscent.fields)
  console.log(`- best ascent layout: visible ${bestAscentSnapshot.visible.filter(Boolean).length}`)
  console.log(renderAscii(bestAscentSnapshot, 56))
  // Hybrid: polish the search winner by gradient.
  const hybrid = ascendFitness(machine, defaultContext, search.top[0]!.fields, {
    steps: 50,
    learningRate: 0.005,
  })
  console.log(`- hybrid (gradient polish of search top): ${search.top[0]!.score.toFixed(3)} -> ${hybrid.score.toFixed(3)}, valid ${hybrid.valid}`)

  // The headline: interpolation with validity projection.
  console.log('\n## Interpolation study')
  const {pairs} = buildStudyPairs(search)
  const studyStart = performance.now()
  const study = runInterpolationStudy(machine, defaultContext, pairs, {
    samplesPerPath: studyConfig.samplesPerPath,
    projection: studyConfig.projection,
  })
  const studyMs = performance.now() - studyStart
  console.log(`- ${study.pairCount} pairs x ${study.pointsPerPath} points = ${study.totalPoints} evaluations in ${studyMs.toFixed(0)}ms`)
  console.log(`- valid before projection: ${study.validBeforeCount}/${study.totalPoints} (${(100 * study.validBeforeFraction).toFixed(1)}%)`)
  console.log(`- valid after projection: ${study.validAfterCount}/${study.totalPoints} (${(100 * study.validAfterFraction).toFixed(1)}%)`)
  console.log(`- points violating each category before projection: overlap ${study.overlapPointCount}, containment ${study.containPointCount}, minimum size ${study.sizePointCount}`)
  console.log(`- fully valid naive paths: ${study.fullyValidPathCount}/${study.pairCount}`)
  console.log(`- repaired ${study.repairedCount}, failed ${study.failedRepairCount}`)
  console.log(`- mean projection steps ${study.meanProjectionSteps.toFixed(1)}, mean moved distance ${study.meanMovedDistance.toFixed(3)} (max ${study.maxMovedDistance.toFixed(3)}) in normalized field space`)
  console.log(`- per-path naive validity: min ${study.pathValidBeforeFractions[0]!.toFixed(2)}, median ${percentile(study.pathValidBeforeFractions, 0.5).toFixed(2)}`)

  // Hero migration.
  console.log('\n## Hero migration (importance mass moves from slot 2 to slot 11)')
  const {a, b} = heroMigrationPair()
  const migration = walkPath(machine, defaultContext, a, b, 21, studyConfig.projection)
  const migrationFlags = migration.points
    .map(point => point.validBefore ? '1' : point.validAfter ? 'P' : 'X')
    .join('')
  console.log(`- validity along 21 points (1 valid, P projected, X failed): ${migrationFlags}`)
  for (const t of [0, 0.5, 1]) {
    const snapshot = snapshotAt(machine, interpolateFields(a, b, t))
    const areas = [2, 11].map(slot =>
      Math.round(snapshot.width[slot]! * snapshot.height[slot]!))
    console.log(`\n### t = ${t} (slot2 area ${areas[0]}, slot11 area ${areas[1]})`)
    console.log(renderAscii(snapshot, 56))
  }

  // Text-aware fitness term via pretext.
  console.log('\n## Text fitness (pretext line breaking, deterministic stub measurer)')
  void loadPretext().then(pretext => {
    if (pretext === null) {
      console.log('- pretext unavailable on this machine; term skipped')
      return
    }
    const prepared = prepareSlotLabels(pretext)
    console.log('| family | text fit | unfitting labels |')
    console.log('| --- | --- | --- |')
    for (const {name, fields} of authoredFamilies()) {
      const snapshot = snapshotAt(machine, fields)
      const textFit = scoreTextFit(pretext, snapshot, prepared)
      const unfitting = textFit.fits
        .map((fit, slot) => !fit && snapshot.visible[slot]! ? slotLabels[slot]! : null)
        .filter(label => label !== null)
      console.log(`| ${name} | ${textFit.value.toFixed(3)} | ${unfitting.join('; ') === '' ? '(none)' : unfitting.join('; ')} |`)
    }
    console.log(`\ntotal wall time ${(performance.now() - totalStart).toFixed(0)}ms`)
  })
}

if (import.meta.main) runAll()
