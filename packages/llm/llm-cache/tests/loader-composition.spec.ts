/** Real Loader composition test: llm-cache boots through cordis.yml and
 * identical requests replay without hitting the adapter. */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'
import LlmRuntime, { LlmAdapter } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import * as llmCache from '../src/index.ts'

let root: string | undefined
let context: Context | undefined

class CountingAdapter extends LlmAdapter {
  requests = 0

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests += 1
    const last = options.messages.at(-1)
    const text = typeof last?.content === 'string' ? last.content : 'ok'
    yield { type: 'content', block: { index: 0, kind: 'text', text: `echo:${text}` } }
    yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
})

async function loadYaml(lines: readonly string[]): Promise<Context> {
  root = await mkdtemp(join(tmpdir(), 'dsh-llm-cache-loader-'))
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [...lines, ''].join('\n'))

  context = new Context()
  context.baseUrl = pathToFileURL(root).href + '/'
  await context.plugin(Loader)
  context.loader.builtins.include = Include
  const modules = new Map<string, unknown>([
    ['@deepseek-ai/dsh-llm', LlmRuntime],
    ['@deepseek-ai/dsh-llm-cache', llmCache],
  ])
  context.loader.internal = {
    version: 'v2',
    async import(specifier: string) {
      if (!modules.has(specifier)) throw new Error(`unexpected Loader import: ${specifier}`)
      return modules.get(specifier)
    },
  } as unknown as NonNullable<typeof context.loader.internal>
  await context.loader.create({
    name: 'cordis:include',
    config: { path: pathToFileURL(configPath).href },
  })
  await context.loader.await()
  return context
}

async function streamOnce(
  ctx: Context,
  text: string,
): Promise<{ out: string; usage?: unknown }> {
  let out = ''
  let usage: unknown
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

describe('real Loader composition', () => {
  it('replays identical requests through the shipping loop without upstream calls',
    { timeout: 60_000 }, async () => {
      const loaded = await loadYaml([
        '- name: \'@deepseek-ai/dsh-llm\'',
        '- name: \'@deepseek-ai/dsh-llm-cache\'',
      ])

      const unloaded = [...loaded.loader.entries()]
        .filter(entry => entry.fiber === undefined && !entry.disabled)
        .map(entry => entry.options.name)
      expect(unloaded).toEqual([])
      expect(loaded.llm).toBeInstanceOf(LlmRuntime)

      const adapter = new CountingAdapter()
      loaded.llm.registerAdapter(['mock'], adapter)

      // First call: miss → upstream.
      const first = await streamOnce(loaded, 'hello')
      expect(first.out).toBe('echo:hello')
      expect(adapter.requests).toBe(1)

      // Identical call: hit → upstream NOT called, same output.
      const second = await streamOnce(loaded, 'hello')
      expect(second.out).toBe('echo:hello')
      expect(adapter.requests).toBe(1)

      // Different call: miss again.
      const third = await streamOnce(loaded, 'other')
      expect(third.out).toBe('echo:other')
      expect(adapter.requests).toBe(2)
    })
})
