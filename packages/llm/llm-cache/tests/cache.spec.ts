/** llm-cache plugin tests: identical requests replay without hitting upstream. */

import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import * as llm from '@deepseek-ai/dsh-llm'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as cache from '../src/index.ts'

const LlmRuntime = llm.default as typeof llm.LlmRuntime
const { LlmAdapter } = llm

/** Adapter that counts how many times it was actually called. */
class CountingAdapter extends LlmAdapter {
  calls = 0

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.calls += 1
    const last = options.messages.at(-1)
    const first = last?.content[0]
    const text = first?.type === 'text'
      ? first.text
      : 'ok'
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: `echo:${text}` }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: `echo:${text}` } }
    yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }
    yield {
      type: 'finish',
      reason: { kind: 'stop' },
    }
  }
}

async function harness() {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  const adapter = new CountingAdapter()
  const cacheFiber = await ctx.plugin(Object.assign(
    (inner: Context) => cache.apply(inner, {}),
    { inject: cache.inject },
  ))
  const disposeAdapter = ctx.llm.registerAdapter(['mock'], adapter)
  return { ctx, adapter, cacheFiber, disposeAdapter }
}

async function call(ctx: Context, text: string): Promise<string> {
  let out = ''
  for await (const chunk of ctx.llm.stream({
    provider: 'mock',
    model: 'mock',
    messages: [createUserMessage({ content: [{ type: 'text', text: text }], source: { kind: 'user' } })],
  })) {
    if (chunk.type === 'block-end' && chunk.block.type === 'text') out += chunk.block.text
  }
  return out
}

describe('llm-cache', () => {
  it('caches identical requests (second call skips upstream)', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    try {
      // First call: miss → upstream called.
      const first = await call(ctx, 'hello')
      expect(first).toBe('echo:hello')
      expect(adapter.calls).toBe(1)

      // Identical call: hit → upstream NOT called, same output.
      const second = await call(ctx, 'hello')
      expect(second).toBe('echo:hello')
      expect(adapter.calls).toBe(1) // still 1 — cache served it

      // Different request: miss → upstream called again.
      const third = await call(ctx, 'different')
      expect(third).toBe('echo:different')
      expect(adapter.calls).toBe(2)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('treats different messages as different keys', async () => {
    const { ctx, adapter, cacheFiber, disposeAdapter } = await harness()
    try {
      await call(ctx, 'a')
      await call(ctx, 'b')
      expect(adapter.calls).toBe(2)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('caches key includes provider and model', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    const adapter = new CountingAdapter()
    const cacheFiber = await ctx.plugin(Object.assign(
      (inner: Context) => cache.apply(inner, {}),
      { inject: cache.inject },
    ))
    const disposeAdapter = ctx.llm.registerAdapter(['mock', 'other'], adapter)

    const callWith = async (provider: string, model: string, text: string): Promise<void> => {
      for await (const _c of ctx.llm.stream({
        provider, model,
        messages: [createUserMessage({ content: [{ type: 'text', text: text }], source: { kind: 'user' } })],
      })) { /* drain */ }
    }

    try {
      await callWith('mock', 'model-a', 'same')
      await callWith('mock', 'model-b', 'same') // different model → miss
      await callWith('other', 'model-a', 'same') // different provider → miss
      expect(adapter.calls).toBe(3)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })

  it('exposes hit/miss stats', async () => {
    const { ctx, cacheFiber, disposeAdapter } = await harness()
    try {
      await call(ctx, 'x')
      await call(ctx, 'x')
      const stats = (cache.apply as unknown as { stat?: () => { hits: number; misses: number; size: number } }).stat?.()
      expect(stats?.hits).toBe(1)
      expect(stats?.misses).toBe(1)
      expect(stats?.size).toBe(1)
    } finally {
      disposeAdapter()
      await cacheFiber.dispose()
    }
  })
})
