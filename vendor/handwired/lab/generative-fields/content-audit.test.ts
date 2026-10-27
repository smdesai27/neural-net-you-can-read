import {expect, test} from 'bun:test'
import {auditContent} from './content-audit.ts'
import {extractSnapshot} from './fitness.ts'
import {authoredFamilies, buildSource, defaultContext, fieldDimension, slotCount} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'

test('historical validity does not imply required content is retained', () => {
  const machine = createFlowLayoutMachine()
  const source = buildSource(authoredFamilies()[3]!.fields, defaultContext)
  const snapshot = extractSnapshot(machine.evaluate(source), machine.output, source)
  expect(snapshot.valid).toBe(true)
  const audit = auditContent(snapshot, Array<boolean>(slotCount).fill(true))
  expect(audit.missingSlots).toHaveLength(10)
  expect(audit.requiredContentRetained).toBe(false)
})

test('strict containment reports overshoot inside the historical wrap allowance', () => {
  const machine = createFlowLayoutMachine()
  const fields = authoredFamilies()[0]!.fields
  fields[fieldDimension('paddingX')] = 0
  fields[fieldDimension('baseSpan')] = 0.23925
  const source = buildSource(fields, defaultContext)
  const snapshot = extractSnapshot(machine.evaluate(source), machine.output, source)
  expect(snapshot.valid).toBe(true)
  const audit = auditContent(snapshot, Array<boolean>(slotCount).fill(true))
  expect(audit.horizontalOverflow).toBeGreaterThan(2)
  expect(audit.strictlyContained).toBe(false)
})
