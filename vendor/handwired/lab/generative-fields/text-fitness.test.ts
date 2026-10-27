import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {authoredFamilies, buildSource, defaultContext} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot} from './fitness.ts'
import {
  labelLineHeight,
  loadPretext,
  prepareSlotLabels,
  scoreTextFit,
  slotLabels,
} from './text-fitness.ts'

// pretext (https://github.com/chenglou/pretext) isn't in the repo: loadPretext() reads
// PRETEXT_SRC, and the tests skip when the variable is unset. Loading it by default would
// install the stub OffscreenCanvas for the whole `bun test lab` process, where the
// text-tabulated and hn-thread tests install fake canvases of their own.
const pretext = await loadPretext()

test.skipIf(pretext === null)('pretext line breaking scores label fit inside the scoring loop', () => {
  const machine = createFlowLayoutMachine()
  const prepared = prepareSlotLabels(pretext!)
  const measured: Record<string, number> = {}
  for (const {name, fields} of authoredFamilies()) {
    const source = buildSource(fields, defaultContext)
    const values = machine.evaluate(source)
    const snapshot = extractSnapshot(values, machine.output, source)
    const textFit = scoreTextFit(pretext!, snapshot, prepared)
    assert.ok(textFit.value >= 0 && textFit.value <= 1)
    measured[name] = textFit.value
  }
  // The README numbers: every family fits its labels except the mosaic, whose
  // smallest slot (slot 15, low importance and low height preference) is too
  // short for even one 20px line after the 12px insets.
  assert.equal(measured['grid'], 1)
  assert.equal(measured['heroRail'], 1)
  assert.equal(measured['list'], 1)
  assert.equal(measured['filmstrip'], 1)
  assert.equal(measured['mosaic'], 15 / 16)
})

test.skipIf(pretext === null)('a wider slot needs fewer lines, monotonically', () => {
  const prepared = pretext!.prepare(slotLabels[2], '14px sans-serif')
  let previousLineCount = Number.POSITIVE_INFINITY
  for (const maxWidth of [60, 120, 240, 480]) {
    const block = pretext!.layout(prepared, maxWidth, labelLineHeight)
    assert.ok(block.lineCount <= previousLineCount)
    assert.equal(block.height, block.lineCount * labelLineHeight)
    previousLineCount = block.lineCount
  }
  assert.equal(previousLineCount, 1)
})
