/** Edge-case tests for llm-cache: TTL expiry, LRU eviction, disabled mode. */

import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import * as llm from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { LruTtlCache, cacheKey } from '../src/index.ts'
import * as cache from '../src/index.ts'

const LlmRuntime = llm.default as typeof llm.LlmRuntime
const { LlmAdapter } = llm

class CountingAdapter extends LlmAdapter {
  calls = 0

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.calls += 1
    const last = options.messages.at(-1)
    const text = typeof last?.content === 'string' ? last.content : 'ok'
    yield { type: 'content', block: { index: 0, kind: 'text', text: `echo:${text}` } }
    yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

async function call(ctx: Context, text: string): Promise<string> {
  let out = ''
  for await (const chunk of ctx.llm.stream({
    provider: 'mock',
    model: 'mock',
    messages: [{ role: 'user', content: text }],
  })) {
    if (chunk.type === 'content' && chunk.block.kind === 'text') out += chunk.block.text
  }
  return out
}

async function harness(config: Record<string, unknown> = {}) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  const adapter = new CountingAdapter()
  const cacheFiber = await ctx.plugin(Object.assign(
    (inner: Context) => cache.apply(inner, config),
    { inject: cache.inject },
  ))
  const disposeAdapter = ctx.llm.registerAdapter(['mock'], adapter)
  return { ctx, adapter, cacheFiber, disposeAdapter }
}

describe('llm-cache edge cases', () => {
  it('LruTtlCache evicts least-recently-used past maxEntries', () => {
    const c = new LruTtlCache(60_000, 2)
    c.set('a', [{ type: 'finish', reason: { kind: 'stop' } }])
    c.set('b', [{ type: 'finish', reason: { kind: 'stop' } }])
    c.get('a') // touch a → b becomes LRU
    c.set('c', [{ type: 'finish', reason: { kind: 'stop' } }])
    expect(c.size).toBe(2)
    expect(c.get('b')).toBeUndefined() // b evicted
    expect(c.get('a')).toBeDefined()
    expect(c.get('c')).toBeDefined()
  })

  it('LruTtlCache expires entries after TTL', () => {
    const c = new LruTtlCache(-1, 10) // negative TTL → already expired
    c.set('x', [{ type: 'finish', reason: { kind: 'stop' } }])
    expect(c.get('x')).toBeUndefined()
  })

  it('LruTtlCache overwrites existing key and keeps size bounded', () => {
    const c = new LruTtlCache(60_000, 2)
    c.set('a', [{ type: 'finish', reason: { kind: 'stop' } }])
    c.set('a', [{ type: 'finish', reason: { kind: 'stop' } }]) // overwrite
    c.set('b', [{ type: 'finish', reason: { kind: 'stop' } }])
    expect(c.size).toBe(2)
    // Touch a so b is LRU, then add c → b evicted (exercises oldest.done=false path).
    c.get('a')
    c.set('c', [{ type: 'finish', reason: { kind: 'stop' } }])
    expect(c.get('b')).toBeUndefined()
    expect(c.get('a')).toBeDefined()
    expect(c.get('c')).toBeDefined()
    // clear() empties the map.
    c.clear()
    expect(c.size).toBe(0)
  })

  it('cacheKey includes tools with stable ordering', () => {
    const base = {
      provider: 'mock',
      model: 'mock',
      messages: [{ role: 'user' as const, content: 'hi' }],
    }
    const k1 = cacheKey({ ...base, tools: [{ name: 'a', params: { x: 1 } }, { name: 'b' }] })
    const k2 = cacheKey({ ...base, tools: [{ name: 'b' }, { name: 'a', params: { x: 1 } }] })
    const k3 = cacheKey({ ...base })
    expect(k1).toBe(k2) // tool order-insensitive
    expect(k1).not.toBe(k3) // tools change the key
  })

  it('disabled mode passes through without caching', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness({ enabled: false })
    try {
      await call(ctx, 'same')
      await call(ctx, 'same')
      expect(adapter.calls).toBe(2) // no caching
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('replay handles usage with missing inputTokens', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    const adapter = new CountingAdapter()
    // Override: emit usage without inputTokens (some providers omit it).
    adapter.stream = async function* (options: GenerateOptions): AsyncIterable<StreamChunk> {
      this.calls += 1
      const last = options.messages.at(-1)
      const text = typeof last?.content === 'string' ? last.content : 'ok'
      yield { type: 'content', block: { index: 0, kind: 'text', text: `echo:${text}` } }
      yield { type: 'usage', usage: { outputTokens: 3 } }
      yield { type: 'finish', reason: { kind: 'stop' } }
    }
    const cacheFiber = await ctx.plugin(Object.assign(
      (inner: Context) => cache.apply(inner, {}),
      { inject: cache.inject },
    ))
    const disposeAdapter = ctx.llm.registerAdapter(['mock'], adapter)
    try {
      await call(ctx, 'hi') // miss, caches
      let usage: { cacheReadTokens?: number } | undefined
      for await (const chunk of ctx.llm.stream({
        provider: 'mock',
        model: 'mock',
        messages: [{ role: 'user', content: 'hi' }],
      })) {
        if (chunk.type === 'usage') usage = chunk.usage
      }
      // Replay: inputTokens=0, cacheReadTokens defaults to 0 (inputTokens undefined).
      expect(usage?.cacheReadTokens).toBe(0)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })
})
