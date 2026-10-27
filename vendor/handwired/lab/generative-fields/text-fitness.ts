// Optional text-aware fitness term backed by pretext (https://github.com/chenglou/pretext):
// prepare() measures label segments once, layout() is pure arithmetic over the
// cached widths, so line breaking runs inside the scoring loop with no DOM.
//
// Bun has no canvas, so prepare() gets a deterministic stub measurer: a fixed
// per-character advance table installed as OffscreenCanvas. The line-breaking
// arithmetic is real pretext; only the advance numbers are synthetic. In a
// browser the same code runs against real canvas measurement.
//
// pretext isn't in the repo: PRETEXT_SRC names a folder holding its src/ files
// (layout.ts and the modules it imports), e.g. lab/ui-exploration/text-tabulated/pretext.
// When the variable is unset, loadPretext() returns null and callers skip the term.

import {resolve} from 'node:path'
import type {LayoutSnapshot} from './fitness.ts'
import {slotCount} from './fields.ts'

type PreparedText = unknown

export type PretextModule = {
  prepare: (text: string, font: string) => PreparedText
  layout: (prepared: PreparedText, maxWidth: number, lineHeight: number) => {
    lineCount: number
    height: number
  }
}

function characterAdvance(character: string, fontSizePixels: number): number {
  if (character === ' ') return 0.28 * fontSizePixels
  if ('iljft.,;:!\''.includes(character)) return 0.28 * fontSizePixels
  if ('mwMW'.includes(character)) return 0.85 * fontSizePixels
  if (character >= 'A' && character <= 'Z') return 0.68 * fontSizePixels
  return 0.52 * fontSizePixels
}

class DeterministicMeasureContext {
  font = '16px sans-serif'
  measureText(text: string) {
    const fontSizeMatch = /(\d+(?:\.\d+)?)\s*px/.exec(this.font)
    const fontSizePixels = fontSizeMatch === null ? 16 : Number.parseFloat(fontSizeMatch[1]!)
    let width = 0
    for (const character of text) width += characterAdvance(character, fontSizePixels)
    return {width}
  }
}

class DeterministicOffscreenCanvas {
  constructor(_width: number, _height: number) {}
  getContext(_kind: string) {
    return new DeterministicMeasureContext()
  }
}

export function installDeterministicMeasurer() {
  const globals = globalThis as Record<string, unknown>
  if (globals['OffscreenCanvas'] === undefined) {
    globals['OffscreenCanvas'] = DeterministicOffscreenCanvas
  }
}

export async function loadPretext(): Promise<PretextModule | null> {
  const pretextSource = process.env['PRETEXT_SRC']
  if (pretextSource === undefined) return null
  installDeterministicMeasurer()
  try {
    return await import(`${resolve(pretextSource)}/layout.ts`) as PretextModule
  } catch (error) {
    throw new Error(`PRETEXT_SRC=${pretextSource} should name a folder holding pretext's src/ files (layout.ts and the modules it imports); importing layout.ts failed: ${String(error)}`)
  }
}

export const slotLabels = [
  'Misty Lake at First Light',
  'Desert Light',
  'Pacific Edge Cliffs and the Long Road South',
  'Midnight City',
  'Forest Path',
  'Alpine Dawn over the Northern Ridge',
  'Amalfi Morning',
  'North Falls',
  'Lavender Hour in the Old Quarter',
  'Concrete Curve',
  'Blue Lagoon',
  'Autumn Road through the Painted Hills',
  'Harbor Fog',
  'Static Meridian',
  'Glasshouse Winter Garden Panorama',
  'Salt Flats',
] as const

export const labelFont = '14px sans-serif'
export const labelLineHeight = 20
export const labelInset = 12

// Fraction of visible slots whose label, line-broken by pretext at the slot's
// inner width, fits within the slot's inner height.
export function scoreTextFit(
  pretext: PretextModule,
  snapshot: LayoutSnapshot,
  prepared: readonly PreparedText[],
): {value: number, fits: boolean[]} {
  const fits: boolean[] = []
  let visibleCount = 0
  let fittingCount = 0
  for (let slot = 0; slot < slotCount; slot++) {
    if (!snapshot.visible[slot]!) {
      fits.push(false)
      continue
    }
    visibleCount++
    const availableWidth = snapshot.width[slot]! - 2 * labelInset
    const availableHeight = snapshot.height[slot]! - 2 * labelInset
    if (availableWidth <= 0 || availableHeight <= 0) {
      fits.push(false)
      continue
    }
    const block = pretext.layout(prepared[slot], availableWidth, labelLineHeight)
    const fit = block.height <= availableHeight
    fits.push(fit)
    if (fit) fittingCount++
  }
  return {value: visibleCount === 0 ? 0 : fittingCount / visibleCount, fits}
}

export function prepareSlotLabels(pretext: PretextModule): PreparedText[] {
  return slotLabels.map(label => pretext.prepare(label, labelFont))
}
