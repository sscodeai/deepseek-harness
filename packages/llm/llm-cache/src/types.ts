/** Types for @deepseek-ai/dsh-llm-cache. */

import type { StreamChunk } from '@deepseek-ai/dsh-llm'

/** Cache hit/miss statistics exposed for tooling and tests. */
export interface LlmCacheStats {
  hits: number
  misses: number
  size: number
}

/** One cached response: the buffered chunks (finish excluded). */
export type CachedResponse = StreamChunk[]
