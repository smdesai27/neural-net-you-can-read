import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {buildSource, defaultContext, makeFieldVector, slotCount} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, selectionScore, type FitnessWeights} from './fitness.ts'
import {ascendFitness} from './optimize.ts'
import {createRandom, percentile, runSearch, sampleFieldVector} from './search.ts'
import {ascentConfig, searchConfig} from './experiments.ts'

const machine = createFlowLayoutMachine()

test('search over 4000 seeded samples finds high-fitness valid layouts, deterministically', () => {
  const search = runSearch(machine, defaultContext, searchConfig)
  // The numbers the README reports. Everything downstream of the seed is
  // FP32/float64 deterministic, so the counts are exact.
  assert.equal(search.sampleCount, 4000)
  assert.equal(search.validCount, 1078)
  // Violation categories across the samples: overlap is structurally
  // unreachable, vertical overflow dominates, minimum sizes fire rarely.
  assert.equal(search.overlapViolationSamples, 0)
  assert.equal(search.containViolationSamples, 2886)
  assert.equal(search.sizeViolationSamples, 44)
  assert.ok(search.top[0]!.snapshot.valid)
  assert.ok(search.top[0]!.fitness > 0.87 && search.top[0]!.fitness < 0.88)
  // Every kept candidate is scored consistently and sorted.
  for (let index = 1; index < search.top.length; index++) {
    assert.ok(search.top[index - 1]!.score >= search.top[index]!.score)
  }
  // Re-running with the same seed reproduces the result exactly.
  const rerun = runSearch(machine, defaultContext, {...searchConfig, keep: 3})
  assert.equal(rerun.validCount, search.validCount)
  assert.equal(rerun.top[0]!.score, search.top[0]!.score)
  assert.equal(rerun.top[0]!.sampleIndex, search.top[0]!.sampleIndex)
})

test('gradient ascent improves all 12 seeded starts to valid layouts', () => {
  const random = createRandom(ascentConfig.startSeed)
  const scores: number[] = []
  let bestScore = Number.NEGATIVE_INFINITY
  let bestFields: Float64Array | null = null
  for (let start = 0; start < ascentConfig.startCount; start++) {
    const startFields = sampleFieldVector(random)
    const startSource = buildSource(startFields, defaultContext)
    const startSnapshot = extractSnapshot(
      machine.evaluate(startSource),
      machine.output,
      startSource,
    )
    const startScore = selectionScore(startSnapshot, scoreFitness(startSnapshot).value)
    const result = ascendFitness(machine, defaultContext, startFields, {
      steps: ascentConfig.steps,
      learningRate: ascentConfig.learningRate,
    })
    assert.ok(result.valid, `start ${start} must end on a valid layout`)
    assert.ok(result.score > startScore + 0.05, `start ${start} must improve substantially`)
    scores.push(result.score)
    if (result.score > bestScore) {
      bestScore = result.score
      bestFields = result.fields
    }
  }
  // Keep the July scores in the historical report, not as optimizer contracts.
  scores.sort((a, b) => a - b)
  assert.ok(scores.at(-1)! > 0.89)
  assert.ok(percentile(scores, 0.5) > 0.8)
  assert.ok(scores[0]! > 0.8)

  // The characterized bias: the best ascent beats search's best (0.875) on
  // the scalar objective while showing few slots. Both facts are the finding;
  // see the README.
  const search = runSearch(machine, defaultContext, {...searchConfig, keep: 1})
  assert.ok(bestScore > search.top[0]!.score)
  const bestSource = buildSource(bestFields!, defaultContext)
  const bestSnapshot = extractSnapshot(machine.evaluate(bestSource), machine.output, bestSource)
  assert.ok(bestSnapshot.visible.filter(Boolean).length < 16)
})

test('without content richness, two visible slabs can receive the maximum score', () => {
  // Demonstrate the underspecified objective without requiring the optimizer
  // to keep reproducing one particular collapse after numerical fixes.
  const preRichnessWeights: FitnessWeights = {
    fill: 0.3,
    verticalFit: 0.2,
    importanceMatch: 0.2,
    balanceX: 0.1,
    sizeComfort: 0.1,
    rowEvenness: 0.1,
    contentRichness: 0,
  }
  const fields = makeFieldVector({
    paddingX: 0, paddingY: 0, gapX: 0, gapY: 0,
    baseSpan: 1, spanFromImportance: 0, baseHeight: 100, heightFromImportance: 400,
    slots: Array.from({length: slotCount}, (_, slot) => ({
      importance: slot === 0 ? 1 : 0.5,
      orderKey: slot / slotCount,
      visibility: slot < 2 ? 1 : 0,
      heightPreference: 0.5,
    })),
  })
  const source = buildSource(fields, defaultContext)
  const snapshot = extractSnapshot(machine.evaluate(source), machine.output, source)
  assert.ok(snapshot.valid)
  assert.equal(snapshot.visible.filter(Boolean).length, 2)
  assert.ok(Math.abs(scoreFitness(snapshot, preRichnessWeights).value - 1) < 1e-8)
})
