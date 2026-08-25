/** Package-owned invariant companion for llm-cache. @module @deepseek-ai/dsh-llm-cache/invariant */

import type { Context } from '@deepseek-ai/cordis'
import type { InvariantInstaller } from '@deepseek-ai/dsh-invariants'

const PACKAGE_NAME = '@deepseek-ai/dsh-llm-cache'

/** Cordis companion plugin name. */
export const name = 'llm-cache-invariant'
/** Service required before the companion can reserve package ownership. */
export const inject = ['invariants']

/**
 * No runtime invariant is currently owned by llm-cache: the cache is a pure
 * in-memory LRU with no durable event stream to validate. The companion
 * registers the package name so ownership is reserved and future invariant
 * checks can attach here.
 */
const install: InvariantInstaller = Object.assign((_ctx: Context) => {
  // No runtime invariant for llm-cache.
}, { inject: ['invariants'] })

/**
 * Register the llm-cache invariant companion.
 * @param ctx - Cordis context carrying the invariant service.
 * @returns the installed registration's disposer after setup succeeds.
 */
export const apply = (ctx: Context): Promise<() => void> =>
  Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
