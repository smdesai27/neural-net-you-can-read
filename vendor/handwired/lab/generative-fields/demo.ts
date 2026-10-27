// Visual demo of the generative fields experiment. Everything on screen is the
// compiled flow layout circuit reading a field vector; this file is only an
// adapter that writes numbers into the source table and projects the output
// geometry onto DOM nodes.
//
// The two stages render the same interpolation position side by side: the left
// stage walks the naive straight line between two field vectors, the right
// stage walks the same line repaired by validity projection (gradient descent
// on the circuit's own violation depth wire, precomputed per transition).
// Family presets, hero migration clicks, seeded search, and the field sliders
// all retarget that transition, so every change of mind is itself an
// interpolation.
//
// Structure follows the repo's gallery demos and chenguini's ui.md: one DOM
// cache built at mount, event handlers that only store raw input, and a single
// requestAnimationFrame render that reads DOM, interprets inputs, evaluates
// the circuit, and batches DOM writes.

import {invariant} from '../../src/runtime/numeric-ui-core.ts'
import {
  authoredFamilies,
  buildSource,
  defaultContext,
  fieldDimension,
  interpolateFields,
  slotCount,
} from './fields.ts'
import {createFlowLayoutMachine} from './flow-layout.ts'
import {extractSnapshot, scoreFitness, type LayoutSnapshot} from './fitness.ts'
import {projectToValidity} from './optimize.ts'
import {runSearch} from './search.ts'
import {slotLabels} from './text-fitness.ts'

// The circuit always runs at its canonical viewport (the context every number
// in the experiment was measured at); the stages scale the result visually.
const stageWorldWidth = defaultContext.viewportWidth
const stageWorldHeight = defaultContext.viewportHeight

// Base interpolation samples per transition; midpoint refinement doubles them.
const transitionBaseSamples = 41
const autoplayLegMs = 5200
const searchSampleCount = 400
const searchKeep = 5
const candidatePreviewScale = 96 / stageWorldWidth

const gapDimension = fieldDimension('gapX')

function importanceDimension(slot: number) {
  return fieldDimension(`slot${slot}.importance`)
}

function visibilityDimension(slot: number) {
  return fieldDimension(`slot${slot}.visibility`)
}

function easeInOut(t: number): number {
  return t * t * (3 - 2 * t)
}

function main() {
  const machine = createFlowLayoutMachine()
  const initialTensorProgram = machine.circuit.tensorProgram
  const families = new Map(authoredFamilies().map(({name, fields}) => [name, fields]))

  function evaluateSnapshot(fields: Float64Array): LayoutSnapshot {
    const source = buildSource(fields, defaultContext)
    const values = machine.evaluate(source)
    return extractSnapshot(values, machine.output, source)
  }

  // A field vector that is already valid passes through; an invalid one is
  // projected back to validity with the circuit's violation depth gradient.
  function projectedPoint(fields: Float64Array): Float64Array {
    const snapshot = evaluateSnapshot(fields)
    if (snapshot.valid) return fields
    return projectToValidity(machine, defaultContext, fields, {}).fields
  }

  type Transition = {
    from: Float64Array
    to: Float64Array
    toName: string
    // Valid field vectors at uniform positions along the repaired line.
    samples: Float64Array[]
    // Sliders edit `to` in place; the samples rebuild lazily before the next
    // walk away from t = 1.
    samplesStale: boolean
  }

  function buildSamples(from: Float64Array, to: Float64Array): Float64Array[] {
    const base: Float64Array[] = []
    for (let index = 0; index < transitionBaseSamples; index++) {
      const t = index / (transitionBaseSamples - 1)
      base.push(projectedPoint(interpolateFields(from, to, t)))
    }
    // One refinement pass: a midpoint of adjacent repaired samples can drift
    // invalid where a wrap decision flips between them; repair those too so
    // the corridor stays valid under the piecewise interpolation the render
    // uses between samples.
    const samples: Float64Array[] = []
    for (let index = 0; index < base.length; index++) {
      samples.push(base[index]!)
      if (index === base.length - 1) continue
      samples.push(projectedPoint(interpolateFields(base[index]!, base[index + 1]!, 0.5)))
    }
    return samples
  }

  function createTransition(from: Float64Array, to: Float64Array, toName: string): Transition {
    return {from, to, toName, samples: buildSamples(from, to), samplesStale: false}
  }

  function fieldsOnPath(transition: Transition, t: number): Float64Array {
    const samples = transition.samples
    if (t <= 0) return samples[0]!
    if (t >= 1) return samples.at(-1)!
    const scaled = t * (samples.length - 1)
    const index = Math.floor(scaled)
    return interpolateFields(samples[index]!, samples[index + 1]!, scaled - index)
  }

  function nearestSampleOnPath(transition: Transition, t: number): Float64Array {
    const samples = transition.samples
    const index = Math.max(0, Math.min(samples.length - 1, Math.round(t * (samples.length - 1))))
    return samples[index]!
  }

  // The visible slot holding the most importance mass, for hero migration.
  function peakSlot(fields: Float64Array): number {
    let peak = 0
    let peakImportance = Number.NEGATIVE_INFINITY
    for (let slot = 0; slot < slotCount; slot++) {
      const visible = fields[visibilityDimension(slot)]! >= 0.5
      const importance = fields[importanceDimension(slot)]!
      if (visible && importance > peakImportance) {
        peakImportance = importance
        peak = slot
      }
    }
    return peak
  }

  type Candidate = {
    fields: Float64Array
    fitness: number
    snapshot: LayoutSnapshot
  }

  // Centralized state; events hold raw input until the next render interprets
  // everything together.
  const st = {
    t: 0,
    autoplay: true,
    autoplayDirection: 1,
    lastFrameTime: null as number | null,
    selectedSlot: 0,
    searchCount: 0,
    searchSummary: 'seeded search over the field space',
    candidates: [] as Candidate[],
    recompilations: 0,
    events: {
      click: null as MouseEvent | null,
      scrub: null as number | null,
      gap: null as number | null,
      importance: null as number | null,
      sliderReleased: false,
    },
  }
  let transition = createTransition(families.get('grid')!, families.get('filmstrip')!, 'filmstrip')
  let transitionFromName = 'grid'

  // DOM cache, built once at mount.
  function requireElement<T extends Element>(selector: string): T {
    const element = document.querySelector<T>(selector)
    invariant(element !== null, `demo element ${selector} is missing`)
    return element
  }

  function createTiles(content: HTMLElement) {
    return Array.from({length: slotCount}, (_, slot) => {
      const tile = document.createElement('div')
      tile.className = `demo-tile tile-${slot}`
      tile.dataset['slot'] = `${slot}`
      const label = document.createElement('span')
      label.textContent = slotLabels[slot]!
      tile.appendChild(label)
      content.appendChild(tile)
      return tile
    })
  }

  const naiveContent = requireElement<HTMLElement>('#stage-naive .stage-content')
  const projectedContent = requireElement<HTMLElement>('#stage-projected .stage-content')
  const domCache = {
    stageNaive: requireElement<HTMLElement>('#stage-naive'), // cache lifetime: mount
    naiveContent, // cache lifetime: mount
    projectedContent, // cache lifetime: mount
    naiveFlash: requireElement<HTMLElement>('#stage-naive .stage-flash'), // cache lifetime: mount
    projectedFlash: requireElement<HTMLElement>('#stage-projected .stage-flash'), // cache lifetime: mount
    naiveTiles: createTiles(naiveContent), // cache lifetime: mount
    projectedTiles: createTiles(projectedContent), // cache lifetime: mount
    naiveValid: requireElement<HTMLElement>('#naive-valid'), // cache lifetime: mount
    naiveDepth: requireElement<HTMLElement>('#naive-depth'), // cache lifetime: mount
    naiveFitness: requireElement<HTMLElement>('#naive-fitness'), // cache lifetime: mount
    projectedValid: requireElement<HTMLElement>('#projected-valid'), // cache lifetime: mount
    projectedDepth: requireElement<HTMLElement>('#projected-depth'), // cache lifetime: mount
    projectedFitness: requireElement<HTMLElement>('#projected-fitness'), // cache lifetime: mount
    statLine: requireElement<HTMLElement>('#stat-line'), // cache lifetime: mount
    recompileLine: requireElement<HTMLElement>('#recompile-line'), // cache lifetime: mount
    playToggle: requireElement<HTMLButtonElement>('#play-toggle'), // cache lifetime: mount
    scrubber: requireElement<HTMLInputElement>('#scrubber'), // cache lifetime: mount
    pathLabel: requireElement<HTMLElement>('#path-label'), // cache lifetime: mount
    searchSummary: requireElement<HTMLElement>('#search-summary'), // cache lifetime: mount
    candidateStrip: requireElement<HTMLElement>('#candidate-strip'), // cache lifetime: per search
    gapSlider: requireElement<HTMLInputElement>('#gap-slider'), // cache lifetime: mount
    importanceSlider: requireElement<HTMLInputElement>('#importance-slider'), // cache lifetime: mount
    importanceName: requireElement<HTMLElement>('#importance-name'), // cache lifetime: mount
  }

  let scheduledFrame: number | null = null
  function scheduleRender() {
    if (scheduledFrame !== null) return
    scheduledFrame = requestAnimationFrame(function renderAndMaybeScheduleAnotherRender(now) {
      scheduledFrame = null
      if (render(now)) scheduleRender()
    })
  }

  // Event handlers only store raw input; render interprets it.
  document.addEventListener('click', event => {
    st.events.click = event
    scheduleRender()
  })
  domCache.scrubber.addEventListener('input', () => {
    st.events.scrub = Number.parseFloat(domCache.scrubber.value)
    scheduleRender()
  })
  domCache.gapSlider.addEventListener('input', () => {
    st.events.gap = Number.parseFloat(domCache.gapSlider.value)
    scheduleRender()
  })
  domCache.importanceSlider.addEventListener('input', () => {
    st.events.importance = Number.parseFloat(domCache.importanceSlider.value)
    scheduleRender()
  })
  for (const slider of [domCache.gapSlider, domCache.importanceSlider]) {
    slider.addEventListener('change', () => {
      st.events.sliderReleased = true
      scheduleRender()
    })
  }
  window.addEventListener('resize', () => scheduleRender())

  function rebuildCandidateStrip(candidates: readonly Candidate[]) {
    // Tiny bounded stateless chunk: wipe and recreate.
    domCache.candidateStrip.textContent = ''
    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index]!
      const preview = document.createElement('button')
      preview.className = 'candidate-preview'
      preview.dataset['candidate'] = `${index}`
      for (let slot = 0; slot < slotCount; slot++) {
        if (!candidate.snapshot.visible[slot]!) continue
        const rectangle = document.createElement('i')
        rectangle.className = `tile-${slot}`
        rectangle.style.left = `${candidate.snapshot.x[slot]! * candidatePreviewScale}px`
        rectangle.style.top = `${candidate.snapshot.y[slot]! * candidatePreviewScale}px`
        rectangle.style.width = `${candidate.snapshot.width[slot]! * candidatePreviewScale}px`
        rectangle.style.height = `${candidate.snapshot.height[slot]! * candidatePreviewScale}px`
        preview.appendChild(rectangle)
      }
      const score = document.createElement('b')
      score.textContent = candidate.fitness.toFixed(3)
      preview.appendChild(score)
      domCache.candidateStrip.appendChild(preview)
    }
  }

  function projectStage(
    tiles: HTMLElement[],
    flash: HTMLElement,
    validNode: HTMLElement,
    depthNode: HTMLElement,
    fitnessNode: HTMLElement,
    snapshot: LayoutSnapshot,
    fitnessValue: number,
    selectedSlot: number,
  ) {
    const bottomLimit = snapshot.viewportHeight - snapshot.paddingY
    for (let slot = 0; slot < slotCount; slot++) {
      const tile = tiles[slot]!
      if (!snapshot.visible[slot]!) {
        tile.style.display = 'none'
        tile.className = `demo-tile tile-${slot}`
        continue
      }
      const offending = !snapshot.valid && (
        snapshot.y[slot]! + snapshot.height[slot]! > bottomLimit + 0.5
        || snapshot.width[slot]! < defaultContext.minVisibleWidth - 0.5
        || snapshot.height[slot]! < defaultContext.minVisibleHeight - 0.5
      )
      const selected = slot === selectedSlot
      tile.style.display = 'block'
      tile.style.transform = `translate3d(${snapshot.x[slot]!}px, ${snapshot.y[slot]!}px, 0)`
      tile.style.width = `${snapshot.width[slot]!}px`
      tile.style.height = `${snapshot.height[slot]!}px`
      tile.className = `demo-tile tile-${slot}${offending ? ' offending' : ''}${selected ? ' selected' : ''}`
    }
    flash.style.opacity = snapshot.valid ? '0' : '1'
    validNode.textContent = snapshot.valid
      ? 'valid'
      : `invalid · ${snapshot.violationCount} violation${snapshot.violationCount === 1 ? '' : 's'}`
    validNode.style.color = snapshot.valid ? 'var(--valid)' : 'var(--invalid)'
    depthNode.textContent = `${snapshot.violationDepth.toFixed(1)}px`
    fitnessNode.textContent = fitnessValue.toFixed(3)
  }

  function render(now: number): boolean {
    // DOM reads, batched.
    const stageWidth = domCache.stageNaive.clientWidth
    const stageHeight = domCache.stageNaive.clientHeight

    // Handle inputs. Locals first; state commits after layout.
    let t = st.t
    let autoplay = st.autoplay
    let autoplayDirection = st.autoplayDirection
    let selectedSlot = st.selectedSlot
    let searchCount = st.searchCount
    let searchSummary = st.searchSummary
    let candidates = st.candidates
    let candidatesChanged = false
    let retargetedTo: {fields: Float64Array, name: string} | null = null

    if (st.events.click !== null) {
      const target = st.events.click.target
      const element = target instanceof Element ? target : null
      if (element !== null) {
        const preset = element.closest<HTMLElement>('[data-preset]')
        const candidate = element.closest<HTMLElement>('[data-candidate]')
        const tile = element.closest<HTMLElement>('[data-slot]')
        if (element.closest('#play-toggle') !== null) {
          autoplay = !autoplay
          if (autoplay && transition.samplesStale) {
            transition = createTransition(transition.from, transition.to, transition.toName)
          }
        } else if (element.closest('#generate') !== null) {
          searchCount++
          const seed = 0x5EED + Math.imul(searchCount, 1_013_904_223)
          const search = runSearch(machine, defaultContext, {
            sampleCount: searchSampleCount,
            seed,
            keep: searchKeep,
          })
          candidates = search.top
            .filter(entry => entry.snapshot.valid)
            .map(entry => ({fields: entry.fields, fitness: entry.fitness, snapshot: entry.snapshot}))
          candidatesChanged = true
          searchSummary =
            `${search.sampleCount} samples · ${search.validCount} valid · best ${search.top[0]!.fitness.toFixed(3)}`
          retargetedTo = {fields: search.top[0]!.fields, name: 'searched #1'}
        } else if (preset !== null) {
          const name = preset.dataset['preset']!
          retargetedTo = {fields: families.get(name)!, name}
        } else if (candidate !== null) {
          const index = Number.parseInt(candidate.dataset['candidate']!, 10)
          retargetedTo = {fields: candidates[index]!.fields, name: `searched #${index + 1}`}
        } else if (tile !== null) {
          // Hero migration: swap the importance mass between the current peak
          // and the clicked slot, then travel there through projected
          // intermediates.
          const slot = Number.parseInt(tile.dataset['slot']!, 10)
          const toFields = new Float64Array(transition.to)
          const peak = peakSlot(toFields)
          const clickedImportance = toFields[importanceDimension(slot)]!
          toFields[importanceDimension(slot)] = toFields[importanceDimension(peak)]!
          toFields[importanceDimension(peak)] = clickedImportance
          retargetedTo = {fields: toFields, name: `hero on ${slotLabels[slot]!.toLowerCase()}`}
          selectedSlot = slot
        }
      }
    }

    if (retargetedTo !== null) {
      const from = transition.samplesStale
        ? transition.to
        : fieldsOnPath(transition, easeInOut(t))
      transitionFromName = transition.toName
      transition = createTransition(from, projectedPoint(retargetedTo.fields), retargetedTo.name)
      t = 0
      autoplay = true
      autoplayDirection = 1
      if (st.events.click !== null) {
        const clickedTile = st.events.click.target instanceof Element
          && st.events.click.target.closest('[data-slot]') !== null
        if (!clickedTile) selectedSlot = peakSlot(transition.to)
      }
    }

    if (st.events.scrub !== null) {
      if (transition.samplesStale) {
        transition = createTransition(transition.from, transition.to, transition.toName)
      }
      t = st.events.scrub
      autoplay = false
    }

    // Sliders edit the destination endpoint in place; the view rests at the
    // destination while dragging, and the corridor rebuilds lazily on the
    // next walk.
    if (st.events.gap !== null || st.events.importance !== null) {
      const toFields = new Float64Array(transition.to)
      if (st.events.gap !== null) toFields[gapDimension] = st.events.gap
      if (st.events.importance !== null) {
        toFields[importanceDimension(selectedSlot)] = st.events.importance
      }
      transition = {...transition, to: toFields, samplesStale: true}
      t = 1
      autoplay = false
    }
    if (st.events.sliderReleased && transition.samplesStale) {
      // On release, snap the edited destination back to validity so resting
      // layouts stay inside the constraints.
      transition = {...transition, to: projectedPoint(transition.to)}
    }

    // Animation tick: ping-pong the interpolation position.
    let stillAnimating = false
    if (autoplay) {
      const last = st.lastFrameTime ?? now
      const delta = Math.min(64, now - last)
      t += autoplayDirection * delta / autoplayLegMs
      if (t >= 1) {
        t = 1
        autoplayDirection = -1
      } else if (t <= 0) {
        t = 0
        autoplayDirection = 1
      }
      stillAnimating = true
    }

    // Layout: run the compiled circuit on both interpolants.
    const tEased = easeInOut(t)
    const naiveSnapshot = evaluateSnapshot(
      interpolateFields(transition.from, transition.to, tEased),
    )
    let projectedSnapshot = evaluateSnapshot(
      transition.samplesStale ? transition.to : fieldsOnPath(transition, tEased),
    )
    if (!projectedSnapshot.valid && !transition.samplesStale) {
      // A wrap decision can flip between two corridor samples, and the layout
      // on the taller side of the flip overflows for a sliver of t no matter
      // how densely the corridor is sampled (the flip is a step, not a ramp).
      // Display the nearest precomputed valid sample instead; the swap is at
      // most half a sample apart and every displayed layout stays a layout the
      // circuit declared valid.
      projectedSnapshot = evaluateSnapshot(nearestSampleOnPath(transition, tEased))
    }
    const naiveFitness = scoreFitness(naiveSnapshot).value
    const projectedFitness = scoreFitness(projectedSnapshot).value
    if (machine.circuit.tensorProgram !== initialTensorProgram) st.recompilations++
    const stageScale = Math.min(
      stageWidth / stageWorldWidth,
      stageHeight / stageWorldHeight,
    )

    // Commit state.
    st.t = t
    st.autoplay = autoplay
    st.autoplayDirection = autoplayDirection
    st.selectedSlot = selectedSlot
    st.searchCount = searchCount
    st.searchSummary = searchSummary
    st.candidates = candidates
    st.lastFrameTime = autoplay ? now : null
    st.events.click = null
    st.events.scrub = null
    st.events.gap = null
    st.events.importance = null
    st.events.sliderReleased = false

    // DOM writes, batched.
    domCache.naiveContent.style.transform = `scale(${stageScale})`
    domCache.projectedContent.style.transform = `scale(${stageScale})`
    projectStage(
      domCache.naiveTiles,
      domCache.naiveFlash,
      domCache.naiveValid,
      domCache.naiveDepth,
      domCache.naiveFitness,
      naiveSnapshot,
      naiveFitness,
      -1,
    )
    projectStage(
      domCache.projectedTiles,
      domCache.projectedFlash,
      domCache.projectedValid,
      domCache.projectedDepth,
      domCache.projectedFitness,
      projectedSnapshot,
      projectedFitness,
      selectedSlot,
    )
    domCache.statLine.textContent =
      `${machine.circuit.operations.length} ops · ${machine.tensorProgram.layers.length} layers · one compiled circuit`
    domCache.recompileLine.textContent = `recompilations ${st.recompilations}`
    domCache.playToggle.textContent = autoplay ? '❚❚' : '▶'
    domCache.scrubber.value = `${t}`
    domCache.pathLabel.textContent = `${transitionFromName} → ${transition.toName}`
    domCache.gapSlider.value = `${transition.to[gapDimension]!}`
    domCache.importanceSlider.value = `${transition.to[importanceDimension(selectedSlot)]!}`
    domCache.importanceName.textContent = slotLabels[selectedSlot]!.toLowerCase()
    if (candidatesChanged) rebuildCandidateStrip(candidates)
    domCache.searchSummary.textContent = searchSummary

    return stillAnimating
  }

  scheduleRender()
}

if (typeof document !== 'undefined') main()
