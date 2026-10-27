// A genome as a short URL-safe string: each gene quantized to 16 bits of its range.

import {type Genome, genes} from '../layout/schema.ts'

export function encodeGenome(genome: Genome): string {
  const bytes = new Uint8Array(genes.length * 2)
  for (let index = 0; index < genes.length; index++) {
    const gene = genes[index]!
    const unit = (genome[index]! - gene.minimum) / (gene.maximum - gene.minimum)
    const level = Math.round(Math.min(1, Math.max(0, unit)) * 65535)
    bytes[index * 2] = level >> 8
    bytes[index * 2 + 1] = level & 255
  }
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
}

export function decodeGenome(text: string): Genome | null {
  try {
    const binary = atob(text.replaceAll('-', '+').replaceAll('_', '/'))
    if (binary.length !== genes.length * 2) return null
    const genome = new Float64Array(genes.length)
    for (let index = 0; index < genes.length; index++) {
      const gene = genes[index]!
      const level = (binary.charCodeAt(index * 2) << 8) | binary.charCodeAt(index * 2 + 1)
      genome[index] = gene.minimum + (gene.maximum - gene.minimum) * (level / 65535)
    }
    return genome
  } catch {
    return null
  }
}
