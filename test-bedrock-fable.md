# Model IDs via Bedrock/Mantle in NanoClaw

> Note: this file was accidentally deleted and reconstructed on 2026-08-05.
> The Fable 5 / Sonnet 5 sections are rebuilt from the original text and may be
> slightly abbreviated. The Opus 5 section was added at reconstruction time.

## Fable 5 and Sonnet 5

The `400 data retention mode 'default' is not available for this model` error
is caused by using a **bare model ID** (`claude-fable-5`) with
`CLAUDE_CODE_USE_MANTLE=1`. The fix is to use the **`anthropic.`-prefixed**
model ID on Bedrock/Mantle:

| Model string tried | `CLAUDE_CODE_USE_MANTLE` | Result |
|---|---|---|
| `claude-fable-5` | `1` | 400 retention error (this bug) |
| `anthropic.claude-fable-5` | `1` | Works |
| `anthropic.claude-fable-5` | unset (legacy `CLAUDE_CODE_USE_BEDROCK` only) | 400 "on-demand throughput isn't supported" — different error, confirms Mantle is required |

An earlier hypothesis ("set both flags") was tested and disproven; the actual
fix is the model-ID prefix.

**Sonnet 5 has no such requirement** — `claude-sonnet-5` and
`anthropic.claude-sonnet-5` both work identically on Bedrock/Mantle.

## Opus 5 (added 2026-08-05)

Switched `ANTHROPIC_MODEL` to Opus 5. Verified against
`https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages`:

| Model string | Result |
|---|---|
| `anthropic.claude-opus-5` | **Works** |
| `us.anthropic.claude-opus-5` | 404 `The model 'us.anthropic.claude-opus-5' does not exist` |
| `claude-opus-5` | 404 does not exist |

The `us.` / `eu.` / `au.` / `global.` prefixes are **bedrock-runtime geo
inference profile IDs only** — Mantle takes the bare `anthropic.`-prefixed ID.
Confirmed by the AWS model card, which lists them in separate endpoint rows.

Specs: 1M context, 128K max output (`max_tokens: 200000` → 400), knowledge
cutoff May 2026.

Parameter surface (probed directly):

| Parameter | Result on Opus 5 |
|---|---|
| no `thinking` field | Works — adaptive thinking is **on** by default |
| `thinking: {type: adaptive}` | Works |
| `thinking: {type: disabled}` | Accepted |
| `thinking: {type: enabled, budget_tokens: N}` | **400** — use adaptive + `output_config.effort` |
| `temperature` | **400** — deprecated for this model |
| `output_config.effort` low/high/xhigh/max | Works |
| `effort: xhigh` + thinking disabled | **400** — effort capped at `high` when thinking is off |

## Harness version requirement

`@anthropic-ai/claude-code` **2.1.206 had no knowledge of `opus-5`** (grep for
`opus-5` in the bundled binary returned nothing), so it fell through the
model-specific launch-effort and tuning tables. Upgraded to **2.1.222**
(agent-sdk 0.3.222), which contains Opus 5 entries including the
`anthropic.claude-opus-5` / `us.anthropic.claude-opus-5` mapping and per-effort
tuning rows. Pin at >= 2.1.222 when running Opus 5.

## Where model IDs are read

`src/container-runner.ts` (`buildLocalEnv`) forwards these `.env` vars to the
agent process when `CLAUDE_CODE_USE_BEDROCK=1` or `CLAUDE_CODE_USE_MANTLE=1`:
`AWS_REGION`, `AWS_BEARER_TOKEN_BEDROCK`, `ANTHROPIC_MODEL`,
`ANTHROPIC_SMALL_FAST_MODEL`, `ANTHROPIC_AWS_WORKSPACE_ID`. In that mode the
credential proxy is bypassed entirely.

Model IDs are not validated or enumerated anywhere else in NanoClaw — set
`ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL` to any valid model ID and it
is passed through. Prefer the `anthropic.`-prefixed form for both.
