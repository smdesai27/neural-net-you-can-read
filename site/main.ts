import {breed, type Candidate} from '../src/breed/breed.ts'
import {decodeGenome, encodeGenome} from '../src/breed/share.ts'
import {type Comparison, emptyTaste, fitTaste, tasteAgreement, type TasteModel} from '../src/breed/taste.ts'
import {stageNames, stages, type Stage} from '../src/layout/labels.ts'
import {contextAt, createRandom, evaluateLayout, readLayout, tasteFeatureNames, tasteFeatures, type Layout} from '../src/layout/run.ts'
import {buildExtendedSource, clampGenome, type Genome, geneIndex, genes, globalField, seedGenome, tileCount} from '../src/layout/schema.ts'
import {
  ablationVersusGradient, eraseProbeDirection, findSeams, geometryDistance, geometryOf, type Importance,
  patchEachLayer, patchEachNamedNeuron, probeLayer, type ProbeData, type ProbeLayer, type Seam, traceAcrossWidths,
} from '../src/lens/experiments.ts'
import {createNetwork} from '../src/lens/network.ts'
import {color, drawLayout, fit, formatNumber, whenPhotosLoad} from './draw.ts'
import {photoUrls} from './photos.ts'
import {createNetworkView, describeNeuron, escapeHtml} from './network-view.ts'

const network = createNetwork()
const {machine} = network
const pageHeight = 800
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T

// ---------------------------------------------------------------- state

const state = {
  genome: seedGenome(),
  width: 1280,
  random: createRandom(Date.now() >>> 0),
  brood: [] as Candidate[],
  picked: [] as number[],
  generation: 0,
  comparisons: [] as Comparison[],
  taste: emptyTaste() as TasteModel,
}

const fromLink = decodeGenome(new URLSearchParams(location.hash.slice(1)).get('g') ?? '')
if (fromLink) state.genome = clampGenome(fromLink)

const sourceFor = (genome: Genome, width = state.width) => buildExtendedSource(genome, contextAt(width, pageHeight))
const layoutFor = (genome: Genome, width = state.width) => evaluateLayout(machine, genome, contextAt(width, pageHeight))
const geneLabel = (name: string) => {
  const slot = /^slot(\d+)\.(\w+)$/.exec(name)
  if (slot) return `tile ${Number(slot[1]) + 1} ${({importance: 'importance', orderKey: 'order', heightPreference: 'height preference'} as Record<string, string>)[slot[2]!] ?? slot[2]}`
  return ({paddingX: 'side padding', paddingY: 'top padding', gapX: 'column gap', gapY: 'row gap', baseSpan: 'base tile width', spanFromImportance: 'width from importance', baseHeight: 'base tile height', heightFromImportance: 'height from importance', minTilePx: 'minimum tile width (px)'} as Record<string, string>)[name] ?? name
}
const later = (() => {
  const timers = new Map<string, number>()
  return (key: string, delay: number, run: () => void) => {
    clearTimeout(timers.get(key))
    timers.set(key, window.setTimeout(run, delay))
  }
})()

for (const element of document.querySelectorAll<HTMLElement>('[data-stat]')) {
  const named = network.labels.filter(label => label?.named).length
  const flowLayers = network.layers.filter((_, layer) => dominantStage(layer) === 'flow').length
  const stat = {neurons: network.neuronCount, layers: network.layers.length, genes: genes.length, named, flowLayers}[element.dataset['stat'] as 'neurons']
  if (stat !== undefined) element.textContent = formatNumber(stat)
}

// ---------------------------------------------------------------- stage

const frame = $('frame')
const tileElements = Array.from({length: tileCount}, (_, index) => {
  const element = document.createElement('div')
  element.className = 'tile'
  element.innerHTML = `<span>${index + 1}</span>`
  element.style.backgroundImage = `url("${photoUrls[index]}")`
  frame.appendChild(element)
  return element
})
const tooltip = $('tooltip')
let stageValues: Float32Array = new Float32Array(0)
let hoveredTile: number | null = null

const mainView = createNetworkView($<HTMLCanvasElement>('network-main'), network, {
  hover(operation, event) {
    if (operation === null) { tooltip.hidden = true; return }
    tooltip.innerHTML = describeNeuron(network, operation, stageValues)
    tooltip.hidden = false
    const wrap = tooltip.parentElement!.getBoundingClientRect()
    const x = Math.min(event.clientX - wrap.left + 14, wrap.width - 330)
    tooltip.style.left = `${Math.max(0, x)}px`
    tooltip.style.top = `${event.clientY - wrap.top + 14}px`
  },
})

const legend = $('legend')
legend.innerHTML = stages.map(stage => `<li style="--swatch: var(--st-${stage})">${stageNames[stage]}</li>`).join('')

function neuronsOfTile(tile: number): Set<number> {
  const set = new Set<number>()
  network.labels.forEach((label, index) => { if (label?.slot === tile) set.add(index) })
  return set
}

function renderStage() {
  const source = sourceFor(state.genome)
  stageValues = network.evaluate(source)
  const layout = readLayout(machine, source, stageValues)
  const available = frame.parentElement!.clientWidth - 24
  const scale = Math.min(1, available / state.width)
  frame.style.width = `${state.width}px`
  frame.style.height = `${pageHeight}px`
  frame.style.transform = `scale(${scale})`
  frame.parentElement!.style.height = `${pageHeight * scale + 24}px`
  frame.classList.toggle('invalid', !layout.valid)
  layout.tiles.forEach((tile, index) => {
    const element = tileElements[index]!
    element.style.left = `${tile.x}px`
    element.style.top = `${tile.y}px`
    element.style.width = `${tile.width}px`
    element.style.height = `${tile.height}px`
  })
  $('width-out').textContent = `${formatNumber(state.width)} px`
  $('stage-status').textContent = `rows ${layout.rows.join('-')} · ${layout.valid ? 'valid' : `${layout.violationCount} rules broken`}`
  mainView.paint({values: stageValues, highlight: hoveredTile === null ? null : neuronsOfTile(hoveredTile)})
}

tileElements.forEach((element, index) => {
  element.addEventListener('mouseenter', () => { hoveredTile = index; mainView.paint({values: stageValues, highlight: neuronsOfTile(index)}) })
  element.addEventListener('mouseleave', () => { hoveredTile = null; mainView.paint({values: stageValues}) })
})

const widthInput = $<HTMLInputElement>('width')
widthInput.addEventListener('input', () => {
  state.width = Number(widthInput.value)
  renderStage()
  renderBrood()
  later('width-lessons', 200, () => { renderBreak(); renderPatch() })
})

// ---------------------------------------------------------------- breeding

const brood = $('brood')
const broodCanvases = Array.from({length: 9}, (_, index) => {
  const figure = document.createElement('figure')
  const canvas = document.createElement('canvas')
  canvas.setAttribute('aria-label', `Child ${index + 1}`)
  const caption = document.createElement('figcaption')
  figure.append(canvas, caption)
  brood.appendChild(figure)
  canvas.addEventListener('click', () => togglePick(index))
  return {canvas, caption}
})

function startOver() {
  state.generation = 0
  state.picked = []
  state.comparisons = []
  state.taste = emptyTaste()
  runBreed([state.genome])
}

function runBreed(parents: Genome[]) {
  const result = breed(machine, {parents, context: contextAt(state.width, pageHeight), strength: Number($<HTMLInputElement>('mutation').value), random: state.random, taste: state.taste})
  state.brood = result.candidates
  state.picked = []
  $('breed-status').textContent = `Generation ${state.generation}. Tried ${result.tried} children at ${formatNumber(state.width)} px; ${result.valid} were valid. Showing ${parents.length === 1 ? 'the parent' : 'both parents'}, ${state.brood.filter(c => !c.parent && !c.wildcard).length} ${state.taste.comparisons ? 'ranked by your taste' : 'unranked (no picks yet)'}, and ${state.brood.filter(c => c.wildcard).length} at random.`
  renderBrood()
}

function renderBrood() {
  state.brood.forEach((candidate, index) => {
    const {canvas, caption} = broodCanvases[index]!
    const layout = layoutFor(candidate.genome)
    candidate.layout = layout
    const {context, width, height} = fit(canvas)
    drawLayout(context, layout, {x: 4, y: 4, width: width - 8, height: height - 8}, {pageWidth: state.width, pageHeight})
    canvas.classList.toggle('picked', state.picked.includes(index))
    const kind = candidate.parent ? 'parent' : candidate.wildcard ? 'wildcard' : 'child'
    caption.textContent = `${kind} · rows ${layout.rows.join('-')}${layout.valid ? '' : ' · invalid at this width'}`
  })
  for (let index = state.brood.length; index < broodCanvases.length; index++) {
    fit(broodCanvases[index]!.canvas)
    broodCanvases[index]!.caption.textContent = ''
  }
}

function togglePick(index: number) {
  if (index >= state.brood.length) return
  $('breed-error').textContent = ''
  const at = state.picked.indexOf(index)
  if (at >= 0) state.picked.splice(at, 1)
  else {
    state.picked.push(index)
    if (state.picked.length > 2) state.picked.shift()
  }
  renderBrood()
}

$('breed-go').addEventListener('click', () => {
  if (state.picked.length === 0) { $('breed-error').textContent = 'Pick one or two layouts first.'; return }
  const winners = state.picked.map(index => state.brood[index]!)
  const losers = state.brood.filter((_, index) => !state.picked.includes(index))
  for (const winner of winners) for (const loser of losers) state.comparisons.push({winner: tasteFeatures(winner.layout), loser: tasteFeatures(loser.layout)})
  state.taste = fitTaste(state.comparisons)
  state.generation++
  renderTaste()
  runBreed(winners.map(winner => winner.genome))
})

$('breed-adopt').addEventListener('click', () => {
  if (state.picked.length === 0) { $('breed-error').textContent = 'Pick a layout to use first.'; return }
  setGenome(new Float64Array(state.brood[state.picked[0]!]!.genome))
})
$('breed-reset').addEventListener('click', () => { state.genome = seedGenome(); onGenomeChange(); startOver() })
const mutationInput = $<HTMLInputElement>('mutation')
mutationInput.addEventListener('input', () => { $('mutation-out').textContent = Number(mutationInput.value).toFixed(1) })
$('share').addEventListener('click', async () => {
  const genome = state.picked.length > 0 ? state.brood[state.picked[0]!]!.genome : state.genome
  const url = `${location.origin}${location.pathname}#g=${encodeGenome(genome)}`
  history.replaceState(null, '', url)
  try {
    await navigator.clipboard.writeText(url)
    $('breed-status').textContent = 'Link copied. It opens with this layout as the main layout.'
  } catch {
    $('breed-status').textContent = `Link: ${url}`
  }
})

function renderTaste() {
  const max = Math.max(1e-9, ...state.taste.weights.map(Math.abs))
  $('taste-bars').innerHTML = tasteFeatureNames.map((name, index) => {
    const weight = state.taste.weights[index]!
    const share = Math.abs(weight) / max * 50
    const left = weight >= 0 ? 50 : 50 - share
    return `<span class="name">${name}</span><span class="track"><span class="zero"></span><span class="fill${weight < 0 ? ' neg' : ''}" style="left:${left}%;width:${share}%"></span></span><span class="num">${weight.toFixed(2)}</span>`
  }).join('')
  $('taste-status').textContent = state.comparisons.length === 0
    ? 'No picks yet.'
    : `${state.comparisons.length} comparisons. The model agrees with ${Math.round(tasteAgreement(state.taste, state.comparisons) * 100)}% of your picks.`
}

// ---------------------------------------------------------------- the main genome

function setGenome(genome: Genome, width?: number) {
  state.genome = clampGenome(genome)
  if (width !== undefined) {
    state.width = Math.round(width)
    widthInput.value = String(state.width)
  }
  history.replaceState(null, '', `#g=${encodeGenome(state.genome)}`)
  onGenomeChange()
}

function onGenomeChange() {
  renderStage()
  later('atlas', 30, renderAtlas)
  later('lessons', 120, () => { renderSeams(); renderBreak(); renderPatch(); markProbesStale() })
}

// ---------------------------------------------------------------- atlas

const atlasGenes = ['slot0.importance', 'minTilePx', 'gapX', 'baseSpan', 'baseHeight', 'spanFromImportance']
const atlasSelect = $<HTMLSelectElement>('atlas-gene')
atlasSelect.innerHTML = atlasGenes.map(name => `<option value="${name}">${geneLabel(name)}</option>`).join('')
const atlasColumns = 48
const atlasRows = 12
const atlasWidths = Array.from({length: atlasColumns}, (_, column) => Math.round(360 + (1920 - 360) * column / (atlasColumns - 1)))
let atlasCells: {genome: Genome, width: number, layout: Layout}[][] = []
let atlasHover: {row: number, column: number} | null = null

function atlasValues(name: string) {
  const gene = genes[geneIndex(name)]!
  return Array.from({length: atlasRows}, (_, row) => gene.minimum + (gene.maximum - gene.minimum) * row / (atlasRows - 1))
}

function renderAtlas() {
  const name = atlasSelect.value
  const index = geneIndex(name)
  const started = performance.now()
  atlasCells = atlasValues(name).map(value => atlasWidths.map(width => {
    const genome = new Float64Array(state.genome)
    genome[index] = value
    return {genome, width, layout: layoutFor(genome, width)}
  }))
  const elapsed = performance.now() - started
  $('atlas-status').textContent = `${atlasRows * atlasColumns} layouts in ${elapsed.toFixed(0)} ms`
  paintAtlas()
}

function paintAtlas() {
  const canvas = $<HTMLCanvasElement>('atlas-canvas')
  canvas.style.height = `${Math.round(canvas.clientWidth * 0.5)}px`
  const {context, width, height} = fit(canvas)
  const cellWidth = width / atlasColumns
  const cellHeight = height / atlasRows
  atlasCells.forEach((row, rowIndex) => row.forEach((cell, columnIndex) => {
    const box = {x: columnIndex * cellWidth + 1, y: rowIndex * cellHeight + 1, width: cellWidth - 2, height: cellHeight - 2}
    drawLayout(context, cell.layout, box, {pageWidth: cell.width, pageHeight, blocks: true})
  }))
  context.strokeStyle = color('accent')
  context.lineWidth = 1.5
  const same = (a: Layout, b: Layout) => a.rows.join() === b.rows.join()
  atlasCells.forEach((row, rowIndex) => row.forEach((cell, columnIndex) => {
    if (columnIndex > 0 && !same(cell.layout, row[columnIndex - 1]!.layout)) {
      context.beginPath(); context.moveTo(columnIndex * cellWidth, rowIndex * cellHeight); context.lineTo(columnIndex * cellWidth, (rowIndex + 1) * cellHeight); context.stroke()
    }
    if (rowIndex > 0 && !same(cell.layout, atlasCells[rowIndex - 1]![columnIndex]!.layout)) {
      context.beginPath(); context.moveTo(columnIndex * cellWidth, rowIndex * cellHeight); context.lineTo((columnIndex + 1) * cellWidth, rowIndex * cellHeight); context.stroke()
    }
  }))
  if (atlasHover) {
    context.strokeStyle = color('ink')
    context.lineWidth = 2
    context.strokeRect(atlasHover.column * cellWidth + 1, atlasHover.row * cellHeight + 1, cellWidth - 2, cellHeight - 2)
  }
  paintAtlasPreview()
}

function paintAtlasPreview() {
  const canvas = $<HTMLCanvasElement>('atlas-preview')
  const {context, width, height} = fit(canvas)
  const cell = atlasHover ? atlasCells[atlasHover.row]?.[atlasHover.column] : undefined
  if (!cell) {
    const layout = layoutFor(state.genome)
    drawLayout(context, layout, {x: 4, y: 4, width: width - 8, height: height - 8}, {pageWidth: state.width, pageHeight})
    $('atlas-readout').textContent = `The main layout at ${formatNumber(state.width)} px: rows ${layout.rows.join('-')}. Hover the atlas to preview other layouts.`
    return
  }
  drawLayout(context, cell.layout, {x: 4, y: 4, width: width - 8, height: height - 8}, {pageWidth: cell.width, pageHeight})
  const name = atlasSelect.value
  const value = cell.genome[geneIndex(name)]!
  $('atlas-readout').textContent = `${formatNumber(cell.width)} px wide, ${geneLabel(name)} ${value < 2 ? value.toFixed(3) : value.toFixed(0)}: rows ${cell.layout.rows.join('-')}, ${cell.layout.valid ? 'valid' : `invalid (${cell.layout.violationCount} rules broken, usually tiles running off the bottom)`}. Click to make it the main layout.`
}

const atlasCanvas = $<HTMLCanvasElement>('atlas-canvas')
const atlasCellAt = (event: MouseEvent) => {
  const rect = atlasCanvas.getBoundingClientRect()
  const column = Math.min(atlasColumns - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * atlasColumns)))
  const row = Math.min(atlasRows - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * atlasRows)))
  return {row, column}
}
atlasCanvas.addEventListener('mousemove', event => {
  const at = atlasCellAt(event)
  if (atlasHover && at.row === atlasHover.row && at.column === atlasHover.column) return
  atlasHover = at
  paintAtlas()
})
atlasCanvas.addEventListener('click', event => {
  const {row, column} = atlasCellAt(event)
  const cell = atlasCells[row]?.[column]
  if (cell) setGenome(new Float64Array(cell.genome), cell.width)
})
atlasSelect.addEventListener('change', renderAtlas)

// ---------------------------------------------------------------- lesson 1: read a neuron

let findValues: Float32Array = new Float32Array(0)
const findTooltip = document.createElement('div')
const findView = createNetworkView($<HTMLCanvasElement>('network-find'), network, {
  hover(operation, event) {
    const canvas = $<HTMLCanvasElement>('network-find')
    if (operation === null) { findTooltip.hidden = true; return }
    findTooltip.className = 'tooltip'
    findTooltip.innerHTML = describeNeuron(network, operation, findValues)
    findTooltip.hidden = false
    if (!findTooltip.parentElement) { canvas.parentElement!.style.position = 'relative'; canvas.parentElement!.appendChild(findTooltip) }
    const wrap = canvas.parentElement!.getBoundingClientRect()
    findTooltip.style.left = `${Math.max(0, Math.min(event.clientX - wrap.left + 14, wrap.width - 330))}px`
    findTooltip.style.top = `${event.clientY - wrap.top + 14}px`
  },
})
const findInput = $<HTMLInputElement>('find')
function renderFind() {
  findValues = stageValues
  const query = findInput.value.trim().toLowerCase()
  if (!query) {
    findView.paint({values: findValues})
    $('find-status').textContent = 'Try "tile 3", "new row", "overlap", or "order".'
    $('find-list').innerHTML = ''
    return
  }
  const matches = new Set<number>()
  const named: number[] = []
  network.labels.forEach((label, index) => {
    if (!label || !label.label.toLowerCase().includes(query)) return
    matches.add(index)
    if (label.named) named.push(index)
  })
  findView.paint({values: findValues, highlight: matches})
  const helpers = matches.size - named.length
  $('find-status').textContent = `${formatNumber(matches.size)} neurons match: ${formatNumber(named.length)} named by the builder${helpers ? `, ${formatNumber(helpers)} helpers named after the neuron they feed` : ''}.`
  $('find-list').innerHTML = named.slice(0, 12).map(index => {
    const value = findValues[network.wireOf(index)]!
    return `<li>${escapeHtml(network.labels[index]!.label)} <code>= ${Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3)}</code> <span class="note">layer ${network.layerOf[index]! + 1}</span></li>`
  }).join('') + (named.length > 12 ? `<li class="note">and ${named.length - 12} more</li>` : '')
}
findInput.addEventListener('input', renderFind)

// ---------------------------------------------------------------- lesson 2: seams

let seams: Seam[] = []
let activeSeam = 0
function renderSeams() {
  seams = findSeams(network, state.genome, 360, 1920, pageHeight)
  activeSeam = Math.min(activeSeam, Math.max(0, seams.length - 1))
  $('seam-list').innerHTML = seams.length === 0
    ? '<li>No seams: this layout keeps its rows at every width.</li>'
    : seams.map((seam, index) => {
      const names = seam.flips.slice(0, 3).map(flip => `<code>${escapeHtml(flip.label)}</code> ${flip.before}→${flip.after}`).join(', ')
      const more = seam.flips.length > 3 ? ` and ${seam.flips.length - 3} more` : ''
      return `<li data-index="${index}"${index === activeSeam ? ' class="on"' : ''}><b>${formatNumber(seam.width, 2)} px</b>: rows ${seam.rowsBefore.join('-')} → ${seam.rowsAfter.join('-')}. Flips ${names}${more}.</li>`
    }).join('')
  plotSeam()
}
$('seam-list').addEventListener('click', event => {
  const item = (event.target as HTMLElement).closest('li[data-index]') as HTMLElement | null
  if (!item) return
  activeSeam = Number(item.dataset['index'])
  for (const element of $('seam-list').querySelectorAll('li')) element.classList.toggle('on', element === item)
  plotSeam()
})

function plotSeam() {
  const canvas = $<HTMLCanvasElement>('seam-plot')
  const {context, width, height} = fit(canvas)
  const seam = seams[activeSeam]
  if (!seam) return
  const flip = seam.flips[0]!
  const scoreWire = network.output.wrapScores[flip.position]!
  const from = Math.max(360, seam.width - 220)
  const to = Math.min(1920, seam.width + 220)
  const points = traceAcrossWidths(network, state.genome, scoreWire, from, to, pageHeight, 400)
  const pad = {left: 44, right: 12, top: 14, bottom: 26}
  const xOf = (w: number) => pad.left + (w - from) / (to - from) * (width - pad.left - pad.right)
  const yOf = (v: number) => pad.top + (1 - v) * (height - pad.top - pad.bottom)
  context.strokeStyle = color('line')
  context.lineWidth = 1
  context.beginPath(); context.moveTo(pad.left, yOf(0.5)); context.lineTo(width - pad.right, yOf(0.5)); context.stroke()
  context.fillStyle = color('muted')
  context.font = '12px system-ui, sans-serif'
  context.fillText('0.5', 12, yOf(0.5) + 4)
  context.fillText('1', 22, yOf(1) + 4)
  context.fillText('0', 22, yOf(0) + 4)
  context.fillText(`${formatNumber(from)} px`, pad.left, height - 8)
  context.fillText(`${formatNumber(to)} px`, width - pad.right - 56, height - 8)
  context.strokeStyle = color('accent')
  context.setLineDash([3, 3])
  context.beginPath(); context.moveTo(xOf(seam.width), pad.top); context.lineTo(xOf(seam.width), height - pad.bottom); context.stroke()
  context.setLineDash([])
  context.strokeStyle = color('ink')
  context.lineWidth = 1.6
  context.beginPath()
  points.forEach((point, index) => {
    const x = xOf(point.width)
    const y = yOf(Math.max(0, Math.min(1, point.value)))
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.stroke()
  $('seam-note').textContent = `The score behind "${network.label(network.output.wrapBits[flip.position]!)!.label}" from ${formatNumber(from)} to ${formatNumber(to)} px. It crosses 0.5 at ${formatNumber(seam.width, 2)} px, the dashed line, and that one crossing changes the rows from ${seam.rowsBefore.join('-')} to ${seam.rowsAfter.join('-')}. The curve bends because the page width enters the network as 1 ÷ width. Wherever another neuron switches nearby, it gets a corner.`
}

// ---------------------------------------------------------------- lesson 3: break it

let importances: Importance[] = []
let breakPoints: {x: number, y: number, item: Importance}[] = []
let breakChoice: Importance | null = null

function renderBreak() {
  const source = sourceFor(state.genome)
  importances = ablationVersusGradient(network, source)
  const behind = importances.filter(item => item.feedsDecision)
  const zero = behind.filter(item => item.gradient === 0).length
  const moving = behind.filter(item => item.ablation > 0.5)
  const largest = Math.max(0, ...moving.map(item => item.ablation))
  $('break-summary').textContent = `${formatNumber(importances.length)} neurons tested. ${formatNumber(behind.length)} sit behind a decision, and ${formatNumber(zero)} of those have gradient exactly 0. Knocking out ${formatNumber(moving.length)} of them still moves tiles, by up to ${formatNumber(largest)} px in total.`
  breakChoice = (breakChoice && importances.find(item => item.wire === breakChoice!.wire))
    ?? [...behind].sort((a, b) => b.ablation - a.ablation)[0] ?? null
  paintBreak()
}

function paintBreak() {
  const canvas = $<HTMLCanvasElement>('break-plot')
  const {context, width, height} = fit(canvas)
  const pad = {left: 52, right: 14, top: 14, bottom: 34}
  const maxGradient = Math.log10(1 + Math.max(1, ...importances.map(item => item.gradient)))
  const maxAblation = Math.log10(1 + Math.max(1, ...importances.map(item => item.ablation)))
  const xOf = (value: number) => pad.left + Math.log10(1 + value) / maxGradient * (width - pad.left - pad.right)
  const yOf = (value: number) => height - pad.bottom - Math.log10(1 + value) / maxAblation * (height - pad.top - pad.bottom)
  context.strokeStyle = color('line')
  context.beginPath(); context.moveTo(pad.left, pad.top); context.lineTo(pad.left, height - pad.bottom); context.lineTo(width - pad.right, height - pad.bottom); context.stroke()
  context.fillStyle = color('muted')
  context.font = '12px system-ui, sans-serif'
  context.fillText('gradient × activation (log) →', pad.left, height - 10)
  context.save(); context.translate(16, height - pad.bottom); context.rotate(-Math.PI / 2); context.fillText('px moved when knocked out (log) →', 0, 0); context.restore()
  breakPoints = importances.map(item => ({x: xOf(item.gradient) + (item.gradient === 0 ? (hash(item.wire) - 0.5) * 8 : 0), y: yOf(item.ablation), item}))
  const accent = color('accent')
  const ink = color('ink')
  for (const point of breakPoints) {
    context.fillStyle = point.item.feedsDecision ? accent : ink
    context.globalAlpha = point.item.feedsDecision ? 0.75 : 0.35
    context.fillRect(point.x - 2, point.y - 2, 4, 4)
  }
  context.globalAlpha = 1
  if (breakChoice) {
    const point = breakPoints.find(p => p.item.wire === breakChoice!.wire)
    if (point) { context.strokeStyle = ink; context.lineWidth = 1.5; context.strokeRect(point.x - 5, point.y - 5, 10, 10) }
  }
  paintBreakPair()
}

function hash(value: number) {
  const x = Math.sin(value * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

function paintBreakPair() {
  const source = sourceFor(state.genome)
  const before = readLayout(machine, source, network.evaluate(source))
  const drawInto = (id: string, layout: Layout) => {
    const {context, width, height} = fit($<HTMLCanvasElement>(id))
    drawLayout(context, layout, {x: 4, y: 4, width: width - 8, height: height - 8}, {pageWidth: state.width, pageHeight})
  }
  drawInto('break-before', before)
  if (!breakChoice) { drawInto('break-after', before); return }
  const values = network.evaluate(source)
  const isDecision = breakChoice.isDecision
  const knocked = isDecision ? 1 - values[breakChoice.wire]! : 0
  const after = readLayout(machine, source, network.evaluate(source, new Map([[breakChoice.wire, knocked]])))
  drawInto('break-after', after)
  $('break-caption').textContent = `Knocked out: ${breakChoice.label} (${isDecision ? 'flipped' : 'set to 0'}). Gradient said ${breakChoice.gradient === 0 ? 'exactly 0' : breakChoice.gradient.toFixed(1)}; tiles moved ${formatNumber(breakChoice.ablation)} px.`
}

$<HTMLCanvasElement>('break-plot').addEventListener('click', event => {
  const rect = (event.target as HTMLCanvasElement).getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top
  let best: {distance: number, item: Importance} | null = null
  for (const point of breakPoints) {
    const distance = (point.x - x) ** 2 + (point.y - y) ** 2
    if (!best || distance < best.distance) best = {distance, item: point.item}
  }
  if (best && best.distance < 400) { breakChoice = best.item; paintBreak() }
})

// ---------------------------------------------------------------- lesson 4: patch it

const patchGenes = ['minTilePx', 'slot0.importance', 'gapX', 'baseHeight', 'slot3.orderKey']
const patchSelect = $<HTMLSelectElement>('patch-gene')
patchSelect.innerHTML = patchGenes.map(name => `<option value="${name}">${geneLabel(name)}</option>`).join('')
const patchAmount = $<HTMLInputElement>('patch-amount')

function patchedGenome(): Genome {
  const index = geneIndex(patchSelect.value)
  const gene = genes[index]!
  const genome = new Float64Array(state.genome)
  genome[index] = genome[index]! + Number(patchAmount.value) * 0.5 * (gene.maximum - gene.minimum)
  return clampGenome(genome)
}

function renderPatch() {
  const amount = Number(patchAmount.value)
  $('patch-amount-out').textContent = `${amount >= 0 ? '+' : '−'}${Math.round(Math.abs(amount) * 50)}% of its range`
  const genomeB = patchedGenome()
  const sourceA = sourceFor(state.genome)
  const sourceB = sourceFor(genomeB)
  const valuesA = network.evaluate(sourceA)
  const valuesB = network.evaluate(sourceB)
  const layoutA = readLayout(machine, sourceA, valuesA)
  const layoutB = readLayout(machine, sourceB, valuesB)
  for (const [id, layout] of [['patch-a', layoutA], ['patch-b', layoutB]] as const) {
    const {context, width, height} = fit($<HTMLCanvasElement>(id))
    drawLayout(context, layout, {x: 4, y: 4, width: width - 8, height: height - 8}, {pageWidth: state.width, pageHeight})
  }
  const distance = geometryDistance(geometryOf(network, valuesA), geometryOf(network, valuesB))
  const bars = $('patch-bars')
  const layersCanvas = $<HTMLCanvasElement>('patch-layers')
  if (distance < 0.5) {
    bars.innerHTML = '<span class="name">This change moves no tile at this width. Try a bigger change or another gene.</span>'
    fit(layersCanvas)
    return
  }
  const neurons = patchEachNamedNeuron(network, sourceA, sourceB).slice(0, 8)
  bars.innerHTML = neurons.map(neuron => {
    const share = Math.max(0, Math.min(1, neuron.effect)) * 100
    return `<span class="name">${escapeHtml(neuron.label)} <span class="note">· layer ${neuron.layer + 1}</span></span><span class="track"><span class="fill" style="left:0;width:${share}%"></span></span><span class="num">${neuron.effect.toFixed(2)}</span>`
  }).join('')
  const effects = patchEachLayer(network, sourceA, sourceB)
  const {context, width, height} = fit(layersCanvas)
  const pad = {left: 40, right: 10, top: 12, bottom: 24}
  const barWidth = (width - pad.left - pad.right) / effects.length
  const yOf = (value: number) => pad.top + (1 - value) / 1.5 * (height - pad.top - pad.bottom)
  context.strokeStyle = color('line')
  context.beginPath(); context.moveTo(pad.left, yOf(0)); context.lineTo(width - pad.right, yOf(0)); context.stroke()
  context.fillStyle = color('muted')
  context.font = '12px system-ui, sans-serif'
  context.fillText('1.0', 10, yOf(1) + 4)
  context.fillText('0', 22, yOf(0) + 4)
  context.fillText('layer 1', pad.left, height - 6)
  context.fillText(`layer ${effects.length}`, width - pad.right - 60, height - 6)
  effects.forEach((effect, layer) => {
    const clamped = Math.max(-0.5, Math.min(1, effect))
    context.fillStyle = color(`st-${dominantStage(layer)}`)
    const top = Math.min(yOf(clamped), yOf(0))
    context.fillRect(pad.left + layer * barWidth, top, Math.max(1, barWidth - 0.5), Math.abs(yOf(clamped) - yOf(0)))
  })
}
patchSelect.addEventListener('change', renderPatch)
patchAmount.addEventListener('input', () => later('patch', 60, renderPatch))

function dominantStage(layer: number): Stage {
  const counts = new Map<Stage, number>()
  for (const operation of network.layers[layer]!) {
    const stage = network.labels[operation]!.stage
    counts.set(stage, (counts.get(stage) ?? 0) + 1)
  }
  return [...counts].sort((a, b) => b[1] - a[1])[0]![0]
}

// ---------------------------------------------------------------- lesson 5: probes

const probeTargets: {name: string, label: string, read: (genome: Genome, source: Float32Array) => number}[] = [
  {name: 'width', label: 'the page width', read: (_, source) => source[globalField.viewportWidth]!},
  {name: 'slot4.heightPreference', label: "tile 5's height preference", read: genome => genome[geneIndex('slot4.heightPreference')]!},
  {name: 'slot0.importance', label: "tile 1's importance", read: genome => genome[geneIndex('slot0.importance')]!},
]
const probeSelect = $<HTMLSelectElement>('probe-target')
probeSelect.innerHTML = probeTargets.map(target => `<option value="${target.name}">${target.label}</option>`).join('')
let probeData: ProbeData | null = null
let probes: ProbeLayer[] = []
let probeChoice: number | null = null
let probeMarks: number[] = []

function markProbesStale() {
  if (probes.length > 0) $('probe-status').textContent = 'The main layout changed. Train again to probe the new one.'
}

$('probe-go').addEventListener('click', () => {
  $('probe-status').textContent = 'Collecting 1,200 layouts and training a probe at every layer…'
  setTimeout(trainProbes, 30)
})
probeSelect.addEventListener('change', () => { if (probes.length) trainProbes() })

function trainProbes() {
  const target = probeTargets.find(entry => entry.name === probeSelect.value)!
  const random = createRandom(0x9E0B)
  const started = performance.now()
  const sources: Float32Array[] = []
  const values: Float32Array[] = []
  const targets: number[] = []
  for (let sample = 0; sample < 1200; sample++) {
    const genome = new Float64Array(state.genome)
    for (let index = 0; index < genes.length; index++) {
      const gene = genes[index]!
      genome[index] = genome[index]! + random.gaussian() * 0.12 * (gene.maximum - gene.minimum)
    }
    clampGenome(genome)
    const source = sourceFor(genome, 600 + random.next() * 1300)
    sources.push(source)
    values.push(network.evaluate(source))
    targets.push(target.read(genome, source))
  }
  probeData = {sources, values, targets}
  probes = network.layers.map((_, layer) => probeLayer(network, probeData!, layer))
  // Erase the most readable layers, best first, until one that moves the tiles
  // and one that moves nothing have turned up.
  const tested = probes.filter(probe => probe.r2 > 0.2).sort((a, b) => b.r2 - a.r2).slice(0, 25)
    .map(probe => ({probe, erased: eraseProbeDirection(network, probeData!, probe)}))
  const used = tested.find(entry => entry.erased.pixelsMovedPerLayout > 1)
  const unused = tested.find(entry => entry.erased.pixelsMovedPerLayout <= 1)
  $('probe-status').textContent = `Trained ${probes.length} probes on ${formatNumber(sources.length)} layouts in ${((performance.now() - started) / 1000).toFixed(1)} s.`
  const describe = ({probe, erased}: {probe: ProbeLayer, erased: {pixelsMovedPerLayout: number, r2After: number}}) =>
    `layer ${probe.layer + 1} reads it with R² ${probe.r2.toFixed(2)}, leaning most on "${strongestNeuron(probe)}"; erasing that direction drops R² to ${erased.r2After.toFixed(2)} and moves the tiles ${formatNumber(erased.pixelsMovedPerLayout)} px per layout`
  $('probe-readout').textContent = [
    used ? `Readable and used: ${describe(used)}.` : 'No layer where erasing it moves the tiles reads it well.',
    unused ? `Readable but unused: ${describe(unused)}.` : `No layer reads ${target.label} well without also using it. Try the page width.`,
  ].join(' ')
  probeMarks = [used?.probe.layer, unused?.probe.layer].filter((layer): layer is number => layer !== undefined)
  probeChoice = null
  paintProbes()
}

// The feature the probe leans on most, by |weight| times the feature's spread.
function strongestNeuron(probe: ProbeLayer): string {
  let best = 0
  probe.features.forEach((_, index) => {
    if (Math.abs(probe.weights[index]! * probe.scale[index]!) > Math.abs(probe.weights[best]! * probe.scale[best]!)) best = index
  })
  const wire = probe.features[best]
  const label = wire === undefined ? null : network.label(wire as Parameters<typeof network.label>[0])
  if (!label) return 'nothing'
  return label.named ? label.label : `a helper for ${label.label}`
}

function paintProbes() {
  const canvas = $<HTMLCanvasElement>('probe-plot')
  const {context, width, height} = fit(canvas)
  if (probes.length === 0) return
  const pad = {left: 40, right: 10, top: 12, bottom: 24}
  const barWidth = (width - pad.left - pad.right) / probes.length
  const yOf = (value: number) => pad.top + (1 - value) * (height - pad.top - pad.bottom)
  context.fillStyle = color('muted')
  context.font = '12px system-ui, sans-serif'
  context.fillText('R² 1', 6, yOf(1) + 4)
  context.fillText('0', 26, yOf(0) + 4)
  context.fillText('layer 1', pad.left, height - 6)
  context.fillText(`layer ${probes.length}`, width - pad.right - 60, height - 6)
  probes.forEach((probe, layer) => {
    context.fillStyle = layer === probeChoice || (probeChoice === null && probeMarks.includes(layer)) ? color('accent') : color(`st-${dominantStage(layer)}`)
    context.fillRect(pad.left + layer * barWidth, yOf(probe.r2), Math.max(1, barWidth - 0.5), yOf(0) - yOf(probe.r2))
  })
}

$<HTMLCanvasElement>('probe-plot').addEventListener('click', event => {
  if (!probeData || probes.length === 0) return
  const rect = (event.target as HTMLCanvasElement).getBoundingClientRect()
  const layer = Math.floor((event.clientX - rect.left - 40) / ((rect.width - 50) / probes.length))
  const probe = probes[layer]
  if (!probe) return
  probeChoice = layer
  paintProbes()
  const erased = eraseProbeDirection(network, probeData, probe)
  $('probe-readout').textContent = `Layer ${layer + 1}: the probe reads ${probeTargets.find(t => t.name === probeSelect.value)!.label} with R² ${probe.r2.toFixed(2)} from ${probe.features.length} neurons, leaning most on "${strongestNeuron(probe)}". Erasing its direction drops R² to ${erased.r2After.toFixed(2)} and moves the tiles ${formatNumber(erased.pixelsMovedPerLayout)} px per layout.`
})

// ---------------------------------------------------------------- start

function renderAll() {
  renderStage()
  renderTaste()
  renderAtlas()
  renderFind()
  renderSeams()
  renderBreak()
  renderPatch()
}

widthInput.value = String(state.width)
startOver()
renderAll()
whenPhotosLoad(() => { renderBrood(); paintAtlasPreview(); paintBreakPair(); renderPatch() })
window.addEventListener('resize', () => later('resize', 100, () => { renderStage(); renderBrood(); paintAtlas(); plotSeam(); paintBreak(); renderPatch(); paintProbes() }))
