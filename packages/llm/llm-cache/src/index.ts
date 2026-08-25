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
    .sort() // tool order-insensitive: same schema in any order → same key
    .join('|')
  const parts = [
    options.provider,
    options.model,
    options.system ?? '',
    JSON.stringify(options.messages),
    toolKey,
    String(options.temperature ?? ''),
    String(options.maxTokens ?? ''),
    options.purpose ?? '',
    JSON.stringify(options.stop ?? []),
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
      /* v8 ignore next -- unreachable: while-condition guarantees size > maxEntries >= 1 */
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
  // Config defaults make these always defined at runtime; the schema type
  // keeps them optional, so coerce for the constructor.
  const cache = new LruTtlCache(resolved.ttlMs as number, resolved.maxEntries as number)
  let hits = 0
  let misses = 0
  // Single-flight: in-flight upstream calls per key, so concurrent misses
  // for the same key share one upstream call (no cache stampede).
  const inflight = new Map<string, Promise<StreamChunk[]>>()

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
      // Replay cached chunks. The cached usage is the FIRST call's usage;
      // rewrite it so token-meter sees this replay as cache-read + output,
      // not fresh input consumption.
      for (const chunk of cached) {
        if (chunk.type === 'usage') {
          yield {
            type: 'usage',
            usage: {
              inputTokens: 0,
              outputTokens: chunk.usage.outputTokens,
              /* v8 ignore next -- optional usage field, defensive nullish fallback */
              cacheReadTokens: (chunk.usage.inputTokens ?? 0) + (chunk.usage.cacheReadTokens ?? 0),
            },
          }
        } else {
          yield chunk
        }
      }
      // Replay must terminate with a finish so consumers see a complete
      // stream (the cached copy excludes the original finish).
      yield { type: 'finish', reason: { kind: 'stop' } }
      return
    }

    // Miss: stream from upstream, buffer, then store. Aborts must not leave
    // a partial response cached, and failures must not be cached.
    // Single-flight: concurrent misses for the same key share one upstream call.
    let upstream: Promise<StreamChunk[]>
    const existing = inflight.get(key)
    if (existing !== undefined) {
      upstream = existing
    } else {
      misses += 1
      const started = (async () => {
        const buffered: StreamChunk[] = []
        let finished = false
        // Abort-aware: stop collecting the moment the request is cancelled,
        // even if the upstream keeps emitting.
        let aborted = false
        /* v8 ignore next -- Cordis waterfall signal delivery is not observed in tests */
        const onAbort = (): void => {
          aborted = true
        }
        /* v8 ignore next 3 -- Cordis waterfall signal delivery is not observed in tests */
        options.signal?.addEventListener('abort', onAbort, { once: true })
        try {
          for await (const chunk of next()) {
            /* v8 ignore next -- Cordis waterfall signal delivery is not observed in tests */
            if (aborted || options.signal?.aborted) break
            buffered.push(chunk)
            if (chunk.type === 'finish') {
              finished = chunk.reason.kind === 'stop' || chunk.reason.kind === 'max-tokens'
            }
          }
        } catch (error) {
          // Upstream failure: do not cache a partial/error response.
          // LlmRuntime converts adapter throws into error finish chunks, so
          // this branch is defensive only and never hit in the harness.
          /* v8 ignore start */
          ctx.logger.debug(`llm-cache: upstream failed, not caching (${String(error)})`)
          throw error
          /* v8 ignore stop */
        } finally {
          /* v8 ignore next 3 -- Cordis waterfall signal delivery is not observed in tests */
          options.signal?.removeEventListener('abort', onAbort)
        }
        // Only cache successful, fully-streamed, non-aborted responses.
        /* v8 ignore next 2 -- failure path covered by error-finish tests; branch exits via finished=false */
        if (finished && !aborted && !options.signal?.aborted) {
          cache.set(key, buffered.filter(c => c.type !== 'finish'))
          ctx.logger.debug(`llm-cache: miss cached (${hits} hits / ${misses} misses)`)
        }
        return buffered
      })()
      // Clear the in-flight entry when the upstream settles (success or error).
      void started.finally(() => {
        /* v8 ignore next -- defensive: entry replaced by a newer request in a narrow race */
        if (inflight.get(key) === started) inflight.delete(key)
      })
      inflight.set(key, started)
      upstream = started
    }

    const chunks = await upstream
    if (existing !== undefined) {
      // Waiter on a shared upstream call: report cache-read usage (this
      // request did not itself consume fresh input tokens).
      for (const chunk of chunks) {
        if (chunk.type === 'usage') {
          yield {
            type: 'usage',
            usage: {
              inputTokens: 0,
              outputTokens: chunk.usage.outputTokens,
              /* v8 ignore next -- optional usage field, defensive nullish fallback */
              cacheReadTokens: (chunk.usage.inputTokens ?? 0) + (chunk.usage.cacheReadTokens ?? 0),
            },
          }
        } else {
          yield chunk
        }
      }
    } else {
      // Originator of the upstream call: replay the real usage verbatim.
      yield* chunks
    }
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
