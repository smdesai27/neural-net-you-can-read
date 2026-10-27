// A 7-weight taste model fit to picks. Each pick of one layout over another is
// one Bradley-Terry comparison: P(winner beats loser) = sigmoid(w . (f_winner -
// f_loser)), where f is the layout's seven upstream fitness terms. The repo's
// generative-fields README lists fitting these weights to human pairwise
// preferences as an untried next step.

export type TasteModel = {weights: number[], comparisons: number}

export type Comparison = {winner: number[], loser: number[]}

export const emptyTaste = (): TasteModel => ({weights: [0, 0, 0, 0, 0, 0, 0], comparisons: 0})

export function scoreTaste(model: TasteModel, features: readonly number[]): number {
  let score = 0
  for (let index = 0; index < model.weights.length; index++) score += model.weights[index]! * features[index]!
  return score
}

// Ridge-regularised logistic regression by gradient descent. Small and
// deterministic: seven weights, at most a few hundred comparisons.
export function fitTaste(comparisons: readonly Comparison[], {ridge = 0.05, steps = 400, rate = 2} = {}): TasteModel {
  const size = 7
  const weights = new Array<number>(size).fill(0)
  if (comparisons.length === 0) return {weights, comparisons: 0}
  const differences = comparisons.map(({winner, loser}) => winner.map((value, index) => value - loser[index]!))
  for (let step = 0; step < steps; step++) {
    const gradient = weights.map(weight => ridge * weight)
    for (const difference of differences) {
      let margin = 0
      for (let index = 0; index < size; index++) margin += weights[index]! * difference[index]!
      const loss = 1 / (1 + Math.exp(margin))
      for (let index = 0; index < size; index++) gradient[index]! -= loss * difference[index]! / differences.length
    }
    for (let index = 0; index < size; index++) weights[index]! -= rate * gradient[index]!
  }
  return {weights, comparisons: comparisons.length}
}

// How often the model ranks the picked layout above the passed-over one.
export function tasteAgreement(model: TasteModel, comparisons: readonly Comparison[]): number {
  if (comparisons.length === 0) return 0
  let agree = 0
  for (const {winner, loser} of comparisons) if (scoreTaste(model, winner) > scoreTaste(model, loser)) agree++
  return agree / comparisons.length
}
