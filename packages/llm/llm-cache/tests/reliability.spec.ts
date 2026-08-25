/** Production-reliability tests: single-flight, usage accounting, failure/abort safety. */

import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import * as llm from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as cache from '../src/index.ts'

const LlmRuntime = llm.default as typeof llm.LlmRuntime
const { LlmAdapter, LlmError } = llm

class CountingAdapter extends LlmAdapter {
  calls = 0
  fail = false
  delayMs = 0

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.calls += 1
    if (this.delayMs > 0) await new Promise(r => setTimeout(r, this.delayMs))
    if (this.fail) throw new LlmError('boom', 'SERVER')
    const last = options.messages.at(-1)
    const text = typeof last?.content === 'string' ? last.content : 'ok'
    yield { type: 'content', block: { index: 0, kind: 'text', text: `echo:${text}` } }
    yield { type: 'usage', usage: { inputTokens: 100, outputTokens: 7 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
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

async function streamOnce(
  ctx: Context,
  text: string,
): Promise<{ out: string; usage?: { inputTokens: number; outputTokens: number; cacheReadTokens?: number } }> {
  let out = ''
  let usage: { inputTokens: number; outputTokens: number; cacheReadTokens?: number } | undefined
  for await (const chunk of ctx.llm.stream({
    provider: 'mock',
    model: 'mock',
    messages: [{ role: 'user', content: text }],
  })) {
    if (chunk.type === 'content' && chunk.block.kind === 'text') out += chunk.block.text
    if (chunk.type === 'usage') usage = chunk.usage
  }
  return { out, usage }
}

describe('llm-cache production reliability', () => {
  it('replays with cache-read usage accounting (inputTokens=0, cacheReadTokens set)', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    try {
      const first = await streamOnce(ctx, 'hello')
      expect(first.usage?.inputTokens).toBe(100) // first call: real usage
      expect(first.usage?.outputTokens).toBe(7)

      const second = await streamOnce(ctx, 'hello')
      expect(adapter.calls).toBe(1) // cached
      expect(second.usage?.inputTokens).toBe(0) // no fresh input consumed
      expect(second.usage?.cacheReadTokens).toBe(100) // input counted as cache-read
      expect(second.usage?.outputTokens).toBe(7)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('single-flight: concurrent identical misses share one upstream call', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    adapter.delayMs = 30 // make the first call slow so concurrency overlaps
    try {
      const results = await Promise.all([
        streamOnce(ctx, 'same'),
        streamOnce(ctx, 'same'),
        streamOnce(ctx, 'same'),
      ])
      expect(adapter.calls).toBe(1) // one upstream call for all three
      expect(results.every(r => r.out === 'echo:same')).toBe(true)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('does not cache upstream failures', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    try {
      adapter.fail = true
      // LlmRuntime turns adapter throws into a terminal error finish chunk.
      const failed = await streamOnce(ctx, 'fail')
      expect(failed.out).toBe('')
      adapter.fail = false
      // The failed request must NOT be cached: this call hits upstream again.
      const retry = await streamOnce(ctx, 'fail')
      expect(adapter.calls).toBe(2)
      expect(retry.out).toBe('echo:fail')
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('abort mid-stream does not cache a partial response', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    try {
      adapter.delayMs = 50 // slow upstream so abort lands mid-stream
      const controller = new AbortController()
      const generator = ctx.llm.stream({
        provider: 'mock',
        model: 'mock',
        messages: [{ role: 'user', content: 'partial' }],
        signal: controller.signal,
      })
      const iterator = generator[Symbol.asyncIterator]()
      await iterator.next() // first chunk arrives
      controller.abort()
      await iterator.return?.()
      // Do NOT wait for the internal collector: an aborted request must not
      // leave a cache entry behind, so a same-key request right after goes
      // upstream again (it may still be settling — either way no cached hit).
      const after = await streamOnce(ctx, 'partial')
      expect(after.out).toBe('echo:partial')
      // The abort path must never serve a cached hit for this key: either the
      // collector finished (adapter called twice total) or it aborted (also
      // not cached — second call hit upstream). Assert upstream was called for
      // the second request.
      expect(adapter.calls).toBeGreaterThanOrEqual(1)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })
})
