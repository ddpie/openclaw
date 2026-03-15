import type { OpenClawConfig } from "../config/config.js";

export const CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000;
export const CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000;

export type ContextWindowSource =
  | "model"
  | "modelsConfig"
  | "knownOverride"
  | "agentContextTokens"
  | "default";

/**
 * Known context window corrections for models whose upstream catalog
 * (pi-ai models.generated.js) has not yet been updated.
 *
 * Claude Opus 4.6 and Sonnet 4.6 support 1M context (GA since 2026-03-13).
 * The pi-ai catalog still reports 200K for most providers.
 * See: https://claude.com/blog/1m-context-ga
 *
 * Overrides are scoped to verified first-party providers only.
 * Proxies/aggregators (openrouter, github-copilot, vercel) are excluded
 * because they may impose their own lower context limits.
 *
 * User modelsConfig overrides always take priority over these corrections.
 * This override can be removed once pi-ai ships the catalog update.
 */
const KNOWN_1M_PROVIDERS = new Set([
  "anthropic",
  "amazon-bedrock",
  "google-vertex",
  "google-antigravity",
  "opencode",
]);

const KNOWN_1M_MODEL_PATTERNS = ["opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6"];

function resolveKnownContextWindowOverride(modelId: string, provider: string): number | null {
  if (!KNOWN_1M_PROVIDERS.has(provider)) {
    return null;
  }
  const lower = modelId.toLowerCase();
  for (const pattern of KNOWN_1M_MODEL_PATTERNS) {
    if (lower.includes(pattern)) {
      return 1_000_000;
    }
  }
  return null;
}

export type ContextWindowInfo = {
  tokens: number;
  source: ContextWindowSource;
};

function normalizePositiveInt(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const int = Math.floor(value);
  return int > 0 ? int : null;
}

export function resolveContextWindowInfo(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelId: string;
  modelContextWindow?: number;
  defaultTokens: number;
}): ContextWindowInfo {
  const fromModelsConfig = (() => {
    const providers = params.cfg?.models?.providers as
      | Record<string, { models?: Array<{ id?: string; contextWindow?: number }> }>
      | undefined;
    const providerEntry = providers?.[params.provider];
    const models = Array.isArray(providerEntry?.models) ? providerEntry.models : [];
    const match = models.find((m) => m?.id === params.modelId);
    return normalizePositiveInt(match?.contextWindow);
  })();
  const fromModel = normalizePositiveInt(params.modelContextWindow);
  const fromKnownOverride = resolveKnownContextWindowOverride(params.modelId, params.provider);
  const baseInfo = fromModelsConfig
    ? { tokens: fromModelsConfig, source: "modelsConfig" as const }
    : fromKnownOverride
      ? { tokens: fromKnownOverride, source: "knownOverride" as const }
      : fromModel
        ? { tokens: fromModel, source: "model" as const }
        : { tokens: Math.floor(params.defaultTokens), source: "default" as const };

  const capTokens = normalizePositiveInt(params.cfg?.agents?.defaults?.contextTokens);
  if (capTokens && capTokens < baseInfo.tokens) {
    return { tokens: capTokens, source: "agentContextTokens" };
  }

  return baseInfo;
}

export type ContextWindowGuardResult = ContextWindowInfo & {
  shouldWarn: boolean;
  shouldBlock: boolean;
};

export function evaluateContextWindowGuard(params: {
  info: ContextWindowInfo;
  warnBelowTokens?: number;
  hardMinTokens?: number;
}): ContextWindowGuardResult {
  const warnBelow = Math.max(
    1,
    Math.floor(params.warnBelowTokens ?? CONTEXT_WINDOW_WARN_BELOW_TOKENS),
  );
  const hardMin = Math.max(1, Math.floor(params.hardMinTokens ?? CONTEXT_WINDOW_HARD_MIN_TOKENS));
  const tokens = Math.max(0, Math.floor(params.info.tokens));
  return {
    ...params.info,
    tokens,
    shouldWarn: tokens > 0 && tokens < warnBelow,
    shouldBlock: tokens > 0 && tokens < hardMin,
  };
}
