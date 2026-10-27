// Character-grid rendering of a layout snapshot, for experiment logs and the
// README. Each visible slot paints its rectangle with its hex digit; cells
// covered by more than one slot show '#', which a valid layout never contains.

import type {LayoutSnapshot} from './fitness.ts'
import {slotCount} from './fields.ts'

export function renderAscii(snapshot: LayoutSnapshot, columns = 64): string {
  const scale = columns / snapshot.viewportWidth
  const rows = Math.max(1, Math.round(snapshot.viewportHeight * scale * 0.5))
  const grid: string[][] = Array.from({length: rows}, () =>
    Array.from({length: columns}, () => '.'))
  for (let slot = 0; slot < slotCount; slot++) {
    if (!snapshot.visible[slot]!) continue
    const left = Math.round(snapshot.x[slot]! * scale)
    const right = Math.round((snapshot.x[slot]! + snapshot.width[slot]!) * scale)
    const top = Math.round(snapshot.y[slot]! * scale * 0.5)
    const bottom = Math.round((snapshot.y[slot]! + snapshot.height[slot]!) * scale * 0.5)
    for (let row = Math.max(0, top); row < Math.min(rows, Math.max(top + 1, bottom)); row++) {
      for (let column = Math.max(0, left); column < Math.min(columns, Math.max(left + 1, right)); column++) {
        grid[row]![column] = grid[row]![column] === '.' ? slot.toString(16) : '#'
      }
    }
  }
  return grid.map(row => row.join('')).join('\n')
}
