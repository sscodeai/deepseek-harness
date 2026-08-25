# @deepseek-ai/dsh-llm-cache

LLM response cache for the DeepSeek Harness. Hooks the `llm/stream` waterfall:
identical requests replay from an in-memory cache instead of hitting the
upstream provider — cutting token spend and latency on repeated calls.

## Model experience

- **Identical requests are served from cache.** Same provider, model, system
  prompt, messages, tools, temperature, and maxTokens → same cache key → the
  response replays from memory. The upstream provider is not called, so no
  tokens are billed and latency drops to ~0.
- **First request for a key is a normal upstream call.** It is buffered and
  stored; the next identical request hits.
- **Different requests never collide.** Provider, model, system, messages,
  tools, temperature, and maxTokens all participate in the cache key.
- **Stale token accounting is avoided.** The terminal `finish` chunk is not
  cached, so a replay never shows another call's token usage. (The consumer
  that requested the replay sees the cached chunks; usage accounting is left
  to the live path / token-meter.)
- **Cache is bounded.** LRU eviction past `maxEntries` (default 1000) and TTL
  expiry (`ttlMs`, default 1 hour) keep memory predictable.

## Install

```sh
pnpm add @deepseek-ai/dsh-llm-cache
```

## Usage

```ts
import { Context } from '@deepseek-ai/cordis'
import * as llmCache from '@deepseek-ai/dsh-llm-cache'

const ctx = new Context()
await ctx.plugin(Object.assign(
  (inner: Context) => llmCache.apply(inner, { ttlMs: 3_600_000, maxEntries: 1000 }),
  { inject: llmCache.inject },
))
```

Or via `cordis.yml`:

```yaml
plugins:
  llm-cache:
    ttlMs: 3600000
    maxEntries: 1000
```

## Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `ttlMs` | number | 3,600,000 | Cache entry TTL in milliseconds |
| `maxEntries` | number | 1000 | Max cached responses (LRU eviction) |
| `enabled` | boolean | true | Set false to bypass the cache |

## Scope

This is **L1-exact** caching: identical requests replay. Semantic (L2) and
prefix-accounting (L3) are future work. The design mirrors fusion-cache (a
framework-agnostic LLM caching middleware) — see
[fusion-cache](https://github.com/om/fusion-cache) for the full three-layer
approach and cost-accounting metrics.

## Development

```sh
pnpm exec tsc --build packages/llm/llm-cache/tsconfig.json   # typecheck
pnpm exec vitest --run packages/llm/llm-cache/tests/          # tests
```
