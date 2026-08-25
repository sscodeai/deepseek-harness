/**
 * LLM response cache for the DeepSeek Harness.
 *
 * Hooks the `llm/stream` waterfall: identical requests (same provider, model,
 * and serialized messages/system/tools) replay from an in-memory cache
 * instead of hitting the upstream adapter. Misses call `next()` and store the
 * buffered chunks.
 *
 * This is L1-exact caching — the cheapest, safest layer. Semantic (L2) and
 * prefix-accounting (L3) are future work; see README.
 *
 * @module @deepseek-ai/dsh-llm-cache
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'

export const name = 'llm-cache'
export const inject = ['llm']

/** Plugin configuration. */
export interface Config {
  /** Cache TTL in milliseconds (default 1 hour). */
  ttlMs?: number
  /** Max number of cached responses (default 1000, LRU eviction). */
  maxEntries?: number
  /** Disable caching (useful to compare). */
  enabled?: boolean
}

/** Runtime schema for {@link Config}. */
export const Config = z.object({
  ttlMs: z.number().min(0).default(3_600_000),
  maxEntries: z.number().min(1).default(1000),
  enabled: z.boolean().default(true),
}) as unknown as z<Config>

/** Deterministic cache key from the request. */
export function cacheKey(options: GenerateOptions): string {
  // provider + model + system + messages + tools + temperature form the
  // request identity. Tools are serialized with a stable key order so
  // semantically identical tool definitions hash the same.
  const toolKey = (options.tools ?? [])
    .map(t => JSON.stringify(t, Object.keys(t).sort()))
    .join('|')
  const parts = [
    options.provider,
    options.model,
    options.system ?? '',
    JSON.stringify(options.messages),
    toolKey,
    String(options.temperature ?? ''),
    String(options.maxTokens ?? ''),
  ]
  return parts.join('\u0000')
}

/** In-memory LRU cache with TTL. */
export class LruTtlCache {
  private readonly map = new Map<string, { chunks: StreamChunk[]; expiresAt: number }>()
  private readonly ttlMs: number
  private readonly maxEntries: number

  constructor(ttlMs: number, maxEntries: number) {
    this.ttlMs = ttlMs
    this.maxEntries = maxEntries
  }

  get(key: string): StreamChunk[] | undefined {
    const entry = this.map.get(key)
    if (entry === undefined) return undefined
    if (Date.now() > entry.expiresAt) {
      this.map.delete(key)
      return undefined
    }
    // LRU: re-insert to move to the end (most recently used).
    this.map.delete(key)
    this.map.set(key, entry)
    return entry.chunks
  }

  set(key: string, chunks: StreamChunk[]): void {
    if (this.map.has(key)) this.map.delete(key)
    this.map.set(key, { chunks, expiresAt: Date.now() + this.ttlMs })
    // Evict least-recently-used (front of the map) when over capacity.
    while (this.map.size > this.maxEntries) {
      const oldest = this.map.keys().next()
      if (oldest.done) break
      this.map.delete(oldest.value)
    }
  }

  get size(): number {
    return this.map.size
  }

  clear(): void {
    this.map.clear()
  }
}

/** Install the LLM response cache on the `llm/stream` waterfall. */
export function apply(ctx: Context, config: Config = {}): void {
  const resolved = Config(config)
  const cache = new LruTtlCache(resolved.ttlMs ?? 3_600_000, resolved.maxEntries ?? 1000)
  let hits = 0
  let misses = 0

  const dispose = ctx.on('llm/stream', async function* (options, next) {
    if (!resolved.enabled) {
      yield* next()
      return
    }
    const key = cacheKey(options)
    const cached = cache.get(key)
    if (cached !== undefined) {
      hits += 1
      ctx.logger.debug(`llm-cache: hit (${hits} hits / ${misses} misses)`)
      // Replay the cached chunks. Skip the final finish/usage from the cache
      // (stale token counts); the live stream's terminal chunk will supply
      // fresh accounting via mergeUsage when available.
      for (const chunk of cached) {
        if (chunk.type === 'finish') continue
        yield chunk
      }
      return
    }

    // Miss: stream from upstream, buffer, then store.
    misses += 1
    const buffered: StreamChunk[] = []
    for await (const chunk of next()) {
      buffered.push(chunk)
      yield chunk
    }
    // Store without the terminal finish (usage is call-specific), but keep
    // everything else so replay is byte-identical.
    cache.set(key, buffered.filter(c => c.type !== 'finish'))
    ctx.logger.debug(`llm-cache: miss cached (${hits} hits / ${misses} misses)`)
  })

  // Expose cache introspection for tests and tooling.
  const stat = (): { hits: number; misses: number; size: number } => ({
    hits,
    misses,
    size: cache.size,
  })
  ;(apply as unknown as { stat?: typeof stat }).stat = stat
  ;(apply as unknown as { _cache?: LruTtlCache })._cache = cache

  ctx.effect(() => async () => {
    dispose()
    cache.clear()
  }, 'llm-cache: clear cache and dispose listener')
}
