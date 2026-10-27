import assert from 'node:assert/strict'
import {test} from 'bun:test'
import {breed, crossover, mutate} from '../src/breed/breed.ts'
import {decodeGenome, encodeGenome} from '../src/breed/share.ts'
import {emptyTaste, fitTaste, tasteAgreement} from '../src/breed/taste.ts'
import {contextAt, createRandom, evaluateLayout} from '../src/layout/run.ts'
import {buildExtendedSource, genes, globalField, seedGenome} from '../src/layout/schema.ts'
import {ablationVersusGradient, eraseProbeDirection, findSeams, patchEachNamedNeuron, probeLayer} from '../src/lens/experiments.ts'
import {createNetwork} from '../src/lens/network.ts'

const network = createNetwork()
const {machine} = network

test('mutation and crossover stay inside every gene range', () => {
  const random = createRandom(1)
  let genome = seedGenome()
  for (let round = 0; round < 200; round++) {
    genome = crossover(mutate(genome, 2.5, random), mutate(seedGenome(), 2.5, random), random)
    genome.forEach((value, index) => assert.ok(value >= genes[index]!.minimum && value <= genes[index]!.maximum))
  }
})

test('breeding keeps the parent and shows only valid children', () => {
  const {candidates, valid} = breed(machine, {parents: [seedGenome()], context: contextAt(1280, 800), strength: 1, random: createRandom(2), taste: emptyTaste()})
  assert.equal(candidates.length, 9)
  assert.equal(candidates[0]!.parent, true)
  assert.ok(valid > 50)
  for (const candidate of candidates) assert.equal(candidate.layout.valid, true)
})

test('the taste model learns a preference planted in the picks', () => {
  const random = createRandom(3)
  const comparisons = Array.from({length: 120}, () => {
    const a = Array.from({length: 7}, () => random.next())
    const b = Array.from({length: 7}, () => random.next())
    return a[0]! > b[0]! ? {winner: a, loser: b} : {winner: b, loser: a}
  })
  const model = fitTaste(comparisons)
  assert.ok(model.weights[0]! > 1, `fill weight ${model.weights[0]}`)
  assert.ok(tasteAgreement(model, comparisons) > 0.9)
})

test('a shared link round-trips the genome to 16 bits per gene', () => {
  const genome = mutate(seedGenome(), 1, createRandom(4))
  const back = decodeGenome(encodeGenome(genome))!
  genome.forEach((value, index) => assert.ok(Math.abs(value - back[index]!) <= (genes[index]!.maximum - genes[index]!.minimum) / 65535 + 1e-9))
  assert.equal(decodeGenome('not a genome'), null)
})

test('seams are where the row structure changes, and each names its flipping neurons', () => {
  const seams = findSeams(network, seedGenome(), 360, 1920, 900)
  assert.ok(seams.length >= 3)
  for (const seam of seams) {
    assert.notDeepEqual(seam.rowsBefore, seam.rowsAfter)
    assert.ok(seam.flips.length > 0 && seam.flips.every(flip => flip.label.includes('starts a new row')))
    const left = evaluateLayout(machine, seedGenome(), contextAt(seam.width - 0.05, 900))
    const right = evaluateLayout(machine, seedGenome(), contextAt(seam.width + 0.05, 900))
    assert.deepEqual(left.rows, seam.rowsBefore)
    assert.deepEqual(right.rows, seam.rowsAfter)
  }
})

test('neurons behind a decision have zero gradient, and some still move tiles when knocked out', () => {
  const results = ablationVersusGradient(network, buildExtendedSource(seedGenome(), contextAt(1280, 800)))
  const behind = results.filter(result => result.feedsDecision)
  assert.ok(behind.length > 100)
  assert.ok(behind.every(result => result.gradient === 0))
  assert.ok(behind.some(result => result.ablation > 100))
})

test('patching the one neuron that reads the minimum tile width carries the whole change', () => {
  const genomeB = seedGenome()
  genomeB[genes.findIndex(gene => gene.name === 'minTilePx')] = 320
  const results = patchEachNamedNeuron(network, buildExtendedSource(seedGenome(), contextAt(1280, 800)), buildExtendedSource(genomeB, contextAt(1280, 800)))
  assert.equal(results[0]!.label, 'narrowest tile, as a fraction of the row')
  assert.ok(Math.abs(results[0]!.effect - 1) < 1e-6)
})

test('the page width is readable in the checks layers, and erasing it there moves nothing', () => {
  const random = createRandom(5)
  const sources: Float32Array[] = []
  const values: Float32Array[] = []
  const targets: number[] = []
  for (let sample = 0; sample < 400; sample++) {
    const source = buildExtendedSource(mutate(seedGenome(), 1, random), contextAt(600 + random.next() * 1300, 800))
    sources.push(source)
    values.push(network.evaluate(source))
    targets.push(source[globalField.viewportWidth]!)
  }
  const data = {sources, values, targets}
  const probes = network.layers.map((_, layer) => probeLayer(network, data, layer))
  const unused = probes.filter(probe => probe.r2 > 0.95).map(probe => ({probe, erased: eraseProbeDirection(network, data, probe)}))
    .find(entry => entry.erased.pixelsMovedPerLayout < 1)
  assert.ok(unused, 'some layer reads the width well without the layout using it there')
  assert.ok(unused.erased.r2After < 0.1)
})
