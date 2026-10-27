// Canvas helpers: sizing for device pixels, theme colors, and drawing a layout.

import type {Layout} from '../src/layout/run.ts'
import {photoUrls} from './photos.ts'

export const photos: HTMLImageElement[] = photoUrls.map(url => {
  const image = new Image()
  image.src = url
  return image
})

export function whenPhotosLoad(callback: () => void) {
  let pending = photos.length
  for (const image of photos) {
    if (image.complete) { if (--pending === 0) callback(); continue }
    image.addEventListener('load', () => { if (--pending === 0) callback() }, {once: true})
    image.addEventListener('error', () => { if (--pending === 0) callback() }, {once: true})
  }
}

export function color(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim() || '#888'
}

// Matches the canvas backing store to its CSS size and returns a 2D context in CSS pixels.
export function fit(canvas: HTMLCanvasElement): {context: CanvasRenderingContext2D, width: number, height: number} {
  const ratio = window.devicePixelRatio || 1
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio)
    canvas.height = Math.round(height * ratio)
  }
  const context = canvas.getContext('2d')!
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, width, height)
  return {context, width, height}
}

function drawCover(context: CanvasRenderingContext2D, image: HTMLImageElement, x: number, y: number, width: number, height: number) {
  if (!image.complete || image.naturalWidth === 0 || width <= 0 || height <= 0) {
    context.fillStyle = color('line')
    context.fillRect(x, y, width, height)
    return
  }
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight)
  const sourceWidth = width / scale
  const sourceHeight = height / scale
  context.drawImage(image, (image.naturalWidth - sourceWidth) / 2, (image.naturalHeight - sourceHeight) / 2, sourceWidth, sourceHeight, x, y, width, height)
}

export type DrawOptions = {
  pageWidth: number
  pageHeight: number
  // Draw tiles as flat blocks instead of photos (cheap, for the atlas).
  blocks?: boolean
  highlightTile?: number
}

// Draws a layout's page scaled to fit the canvas box at (x, y, width, height).
export function drawLayout(context: CanvasRenderingContext2D, layout: Layout, box: {x: number, y: number, width: number, height: number}, options: DrawOptions) {
  const scale = Math.min(box.width / options.pageWidth, box.height / options.pageHeight)
  const pageWidth = options.pageWidth * scale
  const pageHeight = options.pageHeight * scale
  const left = box.x + (box.width - pageWidth) / 2
  const top = box.y + (box.height - pageHeight) / 2
  context.fillStyle = color('bg')
  context.fillRect(left, top, pageWidth, pageHeight)
  context.save()
  context.beginPath()
  context.rect(left, top, pageWidth, pageHeight)
  context.clip()
  layout.tiles.forEach((tile, index) => {
    const x = left + tile.x * scale
    const y = top + tile.y * scale
    const width = tile.width * scale
    const height = tile.height * scale
    if (options.blocks) {
      context.fillStyle = index === 0 ? color('accent') : color('st-flow')
      context.globalAlpha = index === 0 ? 0.85 : 0.55
      context.fillRect(x, y, Math.max(0.5, width - 0.5), Math.max(0.5, height - 0.5))
      context.globalAlpha = 1
    } else {
      drawCover(context, photos[index]!, x, y, width, height)
    }
    if (options.highlightTile === index) {
      context.strokeStyle = color('accent')
      context.lineWidth = 2
      context.strokeRect(x + 1, y + 1, width - 2, height - 2)
    }
  })
  context.restore()
  context.strokeStyle = layout.valid ? color('line') : color('accent')
  context.lineWidth = 1
  context.setLineDash(layout.valid ? [] : [4, 3])
  context.strokeRect(left + 0.5, top + 0.5, pageWidth - 1, pageHeight - 1)
  context.setLineDash([])
  return {left, top, scale}
}

export function formatNumber(value: number, digits = 0): string {
  return value.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits})
}
