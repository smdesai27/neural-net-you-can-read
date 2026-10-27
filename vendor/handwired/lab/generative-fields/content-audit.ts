import type {LayoutSnapshot} from './fitness.ts'
import {slotCount} from './fields.ts'

// The historical circuit permits hiding and a horizontal wrap allowance.
export function auditContent(snapshot: LayoutSnapshot, required: readonly boolean[]) {
  if (required.length !== slotCount) throw new Error('required content must describe every slot')
  const missingSlots: number[] = []
  let horizontalOverflow = 0
  let verticalOverflow = 0
  for (let slot = 0; slot < slotCount; slot++) {
    if (required[slot] && !snapshot.visible[slot]) missingSlots.push(slot)
    if (!snapshot.visible[slot]) continue
    horizontalOverflow = Math.max(horizontalOverflow,
      snapshot.paddingX - snapshot.x[slot]!,
      snapshot.x[slot]! + snapshot.width[slot]! - (snapshot.viewportWidth - snapshot.paddingX),
    )
    verticalOverflow = Math.max(verticalOverflow,
      snapshot.paddingY - snapshot.y[slot]!,
      snapshot.y[slot]! + snapshot.height[slot]! - (snapshot.viewportHeight - snapshot.paddingY),
    )
  }
  return {
    missingSlots,
    horizontalOverflow,
    verticalOverflow,
    requiredContentRetained: missingSlots.length === 0,
    strictlyContained: horizontalOverflow === 0 && verticalOverflow === 0,
  }
}
