// The network drawn as one column per layer and one dot per neuron.

import {type Stage, stages} from '../src/layout/labels.ts'
import type {Network} from '../src/lens/network.ts'
import {color, fit} from './draw.ts'

export type NetworkView = ReturnType<typeof createNetworkView>

export type NetworkPaint = {
  values: Float32Array
  // Neurons to emphasise; everything else dims.
  highlight?: ReadonlySet<number> | null
}

export function createNetworkView(
  canvas: HTMLCanvasElement,
  network: Network,
  handlers: {hover?: (operation: number | null, event: MouseEvent) => void, click?: (operation: number) => void} = {},
) {
  const stageRank = new Map<Stage, number>(stages.map((stage, index) => [stage, index]))
  // Within a layer, neurons are grouped by stage, then tile, then build order.
  const columns = network.layers.map(layer => [...layer].sort((a, b) => {
    const left = network.labels[a]!
    const right = network.labels[b]!
    return stageRank.get(left.stage)! - stageRank.get(right.stage)! || (left.slot ?? 99) - (right.slot ?? 99) || a - b
  }))
  const pad = 10
  // A layer wider than `rowsPerColumn` neurons wraps into several sub-columns,
  // so wide layers read as blocks and the narrow middle reads as a neck.
  const rowsPerColumn = 56
  const blocks = columns.map(column => {
    const subColumns = Math.max(1, Math.ceil(column.length / rowsPerColumn))
    return {subColumns, rows: Math.ceil(column.length / subColumns)}
  })
  const totalSubColumns = blocks.reduce((sum, block) => sum + block.subColumns, 0)
  const firstSubColumn: number[] = []
  blocks.reduce((at, block) => { firstSubColumn.push(at); return at + block.subColumns }, 0)
  let size = {width: 0, height: 0}
  let last: NetworkPaint | null = null

  const pitch = () => ({
    x: (size.width - 2 * pad) / totalSubColumns,
    y: (size.height - 2 * pad) / rowsPerColumn,
  })
  const position = (layer: number, index: number) => {
    const block = blocks[layer]!
    const step = pitch()
    const subColumn = Math.floor(index / block.rows)
    const row = index % block.rows
    const x = pad + (firstSubColumn[layer]! + subColumn + 0.5) * step.x
    const top = pad + (size.height - 2 * pad - block.rows * step.y) / 2
    return {x, y: top + (row + 0.5) * step.y}
  }

  function paint(next: NetworkPaint) {
    last = next
    const {context, width, height} = fit(canvas)
    size = {width, height}
    const stageColor = new Map(stages.map(stage => [stage, color(`st-${stage}`)]))
    const accent = color('accent')
    const step = pitch()
    const dot = Math.max(0.6, Math.min(step.x, step.y) * 0.4)
    const highlight = next.highlight && next.highlight.size > 0 ? next.highlight : null
    for (let layer = 0; layer < columns.length; layer++) {
      const column = columns[layer]!
      for (let index = 0; index < column.length; index++) {
        const operation = column[index]!
        const {x, y} = position(layer, index)
        const isOn = next.values[network.wireOf(operation)]! !== 0
        const isHighlighted = highlight?.has(operation) ?? false
        context.fillStyle = isHighlighted ? accent : stageColor.get(network.labels[operation]!.stage)!
        context.globalAlpha = highlight ? (isHighlighted ? 1 : 0.08) : (isOn ? 0.9 : 0.22)
        const extra = isHighlighted ? 0.8 : 0
        context.fillRect(x - dot - extra, y - dot - extra, (dot + extra) * 2, (dot + extra) * 2)
      }
    }
    context.globalAlpha = 1
  }

  function hit(event: MouseEvent): number | null {
    const rect = canvas.getBoundingClientRect()
    const step = pitch()
    const subColumn = Math.floor((event.clientX - rect.left - pad) / step.x)
    if (subColumn < 0 || subColumn >= totalSubColumns) return null
    let layer = firstSubColumn.findIndex((first, index) => subColumn >= first && subColumn < first + blocks[index]!.subColumns)
    if (layer < 0) return null
    const block = blocks[layer]!
    const top = pad + (size.height - 2 * pad - block.rows * step.y) / 2
    const row = Math.floor((event.clientY - rect.top - top) / step.y)
    if (row < 0 || row >= block.rows) return null
    const index = (subColumn - firstSubColumn[layer]!) * block.rows + row
    return columns[layer]![index] ?? null
  }

  canvas.addEventListener('mousemove', event => handlers.hover?.(hit(event), event))
  canvas.addEventListener('mouseleave', event => handlers.hover?.(null, event))
  canvas.addEventListener('click', event => {
    const operation = hit(event)
    if (operation !== null) handlers.click?.(operation)
  })
  window.addEventListener('resize', () => { if (last) paint(last) })

  return {paint, columns}
}

export function describeNeuron(network: Network, operation: number, values: Float32Array): string {
  const label = network.labels[operation]!
  const kind = network.isDecision[operation] ? 'decision (0 or 1)' : network.circuit.operations[operation]!.kind === 'reglu' ? 'ReLU' : 'weighted sum'
  const value = values[network.wireOf(operation)]!
  const shown = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3)
  const name = label.named ? `<b>${escapeHtml(label.label)}</b>` : `helper for <b>${escapeHtml(label.label)}</b>`
  return `${name}<br>${kind} · layer ${network.layerOf[operation]! + 1} · value ${shown}`
}

export function escapeHtml(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
}
