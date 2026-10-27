import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {buildSource, defaultContext, heroMigrationPair, interpolateFields} from './fields.ts'
import {createFlowLayoutMachine, readSlotGeometry} from './flow-layout.ts'
import {runInterpolationStudy, walkPath} from './interpolation.ts'
import {runSearch} from './search.ts'
import {buildStudyPairs, searchConfig, studyConfig} from './experiments.ts'

const machine = createFlowLayoutMachine()

test('interpolation study: projection repairs every invalid intermediate point', () => {
  const search = runSearch(machine, defaultContext, searchConfig)
  const {pool, pairs} = buildStudyPairs(search)
  assert.equal(pool.length, 5 + studyConfig.poolSearchTop)
  assert.equal(pairs.length, 60)
  const study = runInterpolationStudy(machine, defaultContext, pairs, {
    samplesPerPath: studyConfig.samplesPerPath,
    projection: studyConfig.projection,
  })
  assert.equal(study.totalPoints, 6060)

  // The exact counts the README reports: the study is seeded and FP32/float64
  // deterministic end to end. Naive interpolation breaks most points, and
  // projection against the circuit's own violation depth repairs every one
  // (the README headline: target was 95%, measured 100%).
  assert.equal(study.validBeforeCount, 1906)
  assert.equal(study.repairedCount, 4154)
  assert.equal(study.failedRepairCount, 0)
  assert.equal(study.validAfterCount, 6060)
  assert.ok(study.validAfterFraction >= 0.95)

  // Violation categories among the points before projection: every invalid
  // point is a vertical overflow; overlap and minimum sizes never fire on
  // these paths.
  assert.equal(study.overlapPointCount, 0)
  assert.equal(study.containPointCount, 4154)
  assert.equal(study.sizePointCount, 0)

  // Projection stays cheap and close to the path.
  assert.ok(study.meanProjectionSteps < 10)
  assert.ok(study.meanMovedDistance < 0.35)
}, 30_000)

test('hero migration stays valid at every step while the importance mass moves', () => {
  const {a, b} = heroMigrationPair()
  const path = walkPath(machine, defaultContext, a, b, 21, studyConfig.projection)
  assert.equal(path.validBeforeCount, 21, 'every naive intermediate must already be valid')

  function slotArea(t: number, slot: number) {
    const source = buildSource(interpolateFields(a, b, t), defaultContext)
    const values = machine.evaluate(source)
    const geometry = readSlotGeometry(values, machine.output, slot)
    return geometry.width * geometry.height
  }

  // The hero moves from slot 2 to slot 11: dominant at the ends, equal in the
  // middle, and the transfer is monotone in between.
  assert.ok(slotArea(0, 2) > 5 * slotArea(0, 11))
  assert.ok(slotArea(1, 11) > 5 * slotArea(1, 2))
  assert.ok(Math.abs(slotArea(0.5, 2) - slotArea(0.5, 11)) < 1)
  for (let sample = 1; sample <= 20; sample++) {
    const t = sample / 20
    const previousT = (sample - 1) / 20
    assert.ok(slotArea(t, 2) <= slotArea(previousT, 2) + 1e-6)
    assert.ok(slotArea(t, 11) >= slotArea(previousT, 11) - 1e-6)
  }

  // The measured row relationship at the midpoint: the two equal-area
  // co-heroes sit in separate rows, each sharing its row with one small slot
  // (slot 3 beside slot 2, slot 10 beside slot 11).
  const midSource = buildSource(interpolateFields(a, b, 0.5), defaultContext)
  const midValues = machine.evaluate(midSource)
  const midY = (slot: number) => readSlotGeometry(midValues, machine.output, slot).y
  assert.equal(midY(2), midY(3))
  assert.equal(midY(11), midY(10))
  assert.ok(Math.abs(midY(2) - 111.52) < 0.01)
  assert.ok(Math.abs(midY(11) - 413.92) < 0.01)
  assert.ok(midY(11) - midY(2) > 100, 'the co-heroes do not share or neighbor a row')
})
