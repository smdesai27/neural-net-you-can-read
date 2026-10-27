// Breeding genomes: mutation, crossover, and picking which children to show.

import type {FlowLayoutMachine} from '../layout/circuit.ts'
import {type Layout, type Random, evaluateLayout, tasteFeatures} from '../layout/run.ts'
import {type Genome, type LayoutContext, clampGenome, genes} from '../layout/schema.ts'
import {type TasteModel, scoreTaste} from './taste.ts'

// Genes that belong to one tile move together under crossover, so a child
// inherits a tile's importance, order and height preference from one parent.
const geneGroup = genes.map(gene => {
  const match = /^slot(\d+)\./.exec(gene.name)
  return match ? `slot${match[1]}` : gene.name
})

export function mutate(genome: Genome, strength: number, random: Random): Genome {
  const child = new Float64Array(genome)
  for (let index = 0; index < genes.length; index++) {
    const gene = genes[index]!
    let step = random.gaussian() * (gene.maximum - gene.minimum) * 0.06 * strength
    if (random.next() < 0.05 * strength) step *= 4
    child[index] = child[index]! + step
  }
  return clampGenome(child)
}

export function crossover(a: Genome, b: Genome, random: Random): Genome {
  const fromA = new Map<string, boolean>()
  const child = new Float64Array(a.length)
  for (let index = 0; index < genes.length; index++) {
    const group = geneGroup[index]!
    if (!fromA.has(group)) fromA.set(group, random.next() < 0.5)
    child[index] = fromA.get(group) ? a[index]! : b[index]!
  }
  return child
}

export type Candidate = {genome: Genome, layout: Layout, taste: number, wildcard: boolean, parent: boolean}

export type BreedOptions = {
  parents: Genome[]
  context: LayoutContext
  strength: number
  random: Random
  taste: TasteModel
  // How many children to try before choosing which to show.
  tries?: number
  shown?: number
  wildcards?: number
}

// Keeps the parents, tries `tries` valid-or-not children, drops invalid ones,
// then fills the grid with the children taste ranks highest plus a few random
// ones so breeding keeps exploring.
export function breed(machine: FlowLayoutMachine, options: BreedOptions): {candidates: Candidate[], tried: number, valid: number} {
  const {parents, context, strength, random, taste} = options
  const tries = options.tries ?? 256
  const shown = options.shown ?? 9
  const wildcards = options.wildcards ?? 2
  const kept: Candidate[] = parents.map(genome => {
    const layout = evaluateLayout(machine, genome, context)
    return {genome, layout, taste: scoreTaste(taste, tasteFeatures(layout)), wildcard: false, parent: true}
  })
  const pool: Candidate[] = []
  for (let attempt = 0; attempt < tries; attempt++) {
    const base = parents.length > 1
      ? crossover(parents[0]!, parents[1 + Math.floor(random.next() * (parents.length - 1))]!, random)
      : parents[0]!
    const genome = mutate(base, strength, random)
    const layout = evaluateLayout(machine, genome, context)
    if (!layout.valid) continue
    pool.push({genome, layout, taste: scoreTaste(taste, tasteFeatures(layout)), wildcard: false, parent: false})
  }
  const valid = pool.length
  pool.sort((a, b) => b.taste - a.taste)
  const slots = shown - kept.length
  const ranked = pool.splice(0, Math.max(0, slots - wildcards))
  const random_ = pool.sort(() => random.next() - 0.5).slice(0, slots - ranked.length).map(candidate => ({...candidate, wildcard: true}))
  return {candidates: [...kept, ...ranked, ...random_], tried: tries, valid}
}
