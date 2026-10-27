// Prefix networks: fixed arrangements of pairwise combines that compute every prefix items[0],
// combine(items[0], items[1]), combine(combine(items[0], items[1]), items[2]), ... with a far shorter longest chain of
// combines than a loop. combine(earlier, later) has to be associative, i.e. combine(combine(a, b), c) has to give the
// same result as combine(a, combine(b, c)). That holds exactly for selections, e.g. reading tables of integer states,
// and for sums of integers below 2^24, while other FP32 sums and products hold it only up to rounding, where the last
// bits depend on the grouping. combine doesn't have to be commutative. E.g. every row's bottom from the row heights:
// brentKungPrefix(heights, (earlier, later) => circuit.roundAffine([[earlier, 1], [later, 1]])). docs/patterns.md's
// "Running totals" and "A state machine over queued events" explain the costs.

// Sklansky (1960): for blocks of 2, 4, 8, ... items, combine the last prefix of the block's left half into every prefix
// of its right half. For k items, a power of two, that's (k / 2) × log2(k) combines, and a longest chain of log2(k).
export function sklanskyPrefix<T>(items: readonly T[], combine: (earlier: T, later: T) => T): T[] {
  const prefixes = [...items]
  for (let halfSize = 1; halfSize < prefixes.length; halfSize *= 2) {
    for (let blockStart = 0; blockStart < prefixes.length; blockStart += 2 * halfSize) {
      const leftHalfLast = blockStart + halfSize - 1
      const blockEnd = Math.min(blockStart + 2 * halfSize, prefixes.length)
      for (let index = leftHalfLast + 1; index < blockEnd; index++) {
        prefixes[index] = combine(prefixes[leftHalfLast]!, prefixes[index]!)
      }
    }
  }
  return prefixes
}

// Brent-Kung (1982): an upward pass combines neighbors, then pairs of pairs, and so on, which completes the prefixes of
// the first 2, 4, 8, ... items. A downward pass then completes every other prefix by combining one completed prefix
// with one smaller block. For k items, a power of two and at least 4, that's 2k − 2 − log2(k) combines, and a longest
// chain of 2 × log2(k) − 2.
export function brentKungPrefix<T>(items: readonly T[], combine: (earlier: T, later: T) => T): T[] {
  const prefixes = [...items]
  let span = 1
  for (; span < prefixes.length; span *= 2) {
    for (let index = 2 * span - 1; index < prefixes.length; index += 2 * span) {
      prefixes[index] = combine(prefixes[index - span]!, prefixes[index]!)
    }
  }
  for (span /= 2; span >= 1; span /= 2) {
    for (let index = 3 * span - 1; index < prefixes.length; index += 2 * span) {
      prefixes[index] = combine(prefixes[index - span]!, prefixes[index]!)
    }
  }
  return prefixes
}
