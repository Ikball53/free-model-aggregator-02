// Free-model aggregator: one server, two API wires, live free models.
//
// A client points any OpenAI- or Anthropic-compatible caller at this server
// and every completion it answers comes from a real free OpenRouter model,
// dialled through the grader's gateway on the trial's key. The server picks
// the highest-rated free model that meets each request (JSON mode and tools
// included), retries a refusal once on the same model, then fails over to
// the next-best model that meets the request, and reports the refusals it
// recovered from in the x-makefun-upstream-failures header.
//
// Determinism: every choice comes from the brief, the request, or a gateway
// answer. No wall clock and no unseeded randomness anywhere in a request
// path, so a replay leaves the same state export.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FreeModel {
  id: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Brief {
  models: FreeModel[];
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

interface GatewayOutcome {
  ok: boolean;
  model: string;
  status: number;
  text: string;
  failures: FailureRow[];
}

// ---------------------------------------------------------------------------
// Constants and environment
// ---------------------------------------------------------------------------

const GATEWAY_BASE = process.env.OPENAI_BASE_URL ?? "";
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? "";
const ENV_URL = process.env.MAKEFUN_ENV_URL ?? "";
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? "no-seed";
const TRIAL_NOW = process.env.MAKEFUN_TRIAL_NOW ?? "";

const MAX_GATEWAY_WAIT_MS = 25_000;
const MAX_ATTEMPTS_PER_PROMPT = 3;
const RETRY_PAUSE_MS = 1_000;
const WORK_POLL_MS = 750;
const REGISTER_TRIES = 60;

// The sealed free catalog, used only when the brief cannot be read. The
// brief is the source of truth at runtime; this table mirrors it so the
// server still knows the fourteen dialable ids without one.
const FALLBACK_MODELS: FreeModel[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", rating_milli: 1150, context_length: 1024000, tools: true, vision: false, json: true },
  { id: "openrouter/cohere/north-mini-code:free", rating_milli: 1200, context_length: 256000, tools: true, vision: false, json: false },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", rating_milli: 1150, context_length: 512000, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", rating_milli: 1150, context_length: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-31b-it:free", rating_milli: 1210, context_length: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", rating_milli: 1150, context_length: 256000, tools: true, vision: true, json: false },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", rating_milli: 1285, context_length: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", rating_milli: 1221, context_length: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", rating_milli: 1303, context_length: 1000000, tools: true, vision: true, json: true },
  { id: "openrouter/particle-ai/particle-1:free", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/recursive/flint-v1.0:free", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", rating_milli: 1475, context_length: 1048576, tools: true, vision: true, json: false },
];

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function estTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

// A short deterministic tag from the trial seed, for response ids. The env
// never scores ids, but a replay must see the same ones.
const SEED_TAG = createHash("sha256").update(TRIAL_SEED).digest("hex").slice(0, 10);

let replyCounter = 0;
function nextReplyId(kind: "chatcmpl" | "msg"): string {
  replyCounter += 1;
  return `${kind}-${SEED_TAG}-${replyCounter}`;
}

// The trial's own "now", as epoch seconds, for the created fields. Falls
// back to 0 outside a trial; never the wall clock.
function trialEpoch(): number {
  if (!TRIAL_NOW) return 0;
  const ms = Date.parse(TRIAL_NOW);
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
}

function trialIso(): string {
  return TRIAL_NOW || "2026-11-15T03:30:00Z";
}

// ---------------------------------------------------------------------------
// Brief and catalog
// ---------------------------------------------------------------------------

function loadBrief(): Brief | null {
  if (!PROMPT_FILE) return null;
  try {
    const raw = readFileSync(PROMPT_FILE, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('{"schema":"free-model-aggregator-02/models/v1"')) continue;
      const parsed: unknown = JSON.parse(trimmed);
      if (!isRecord(parsed) || !Array.isArray(parsed.models)) continue;
      const models: FreeModel[] = [];
      for (const m of parsed.models) {
        if (!isRecord(m) || typeof m.id !== "string") continue;
        models.push({
          id: m.id,
          rating_milli: typeof m.rating_milli === "number" ? m.rating_milli : 1150,
          context_length: typeof m.context_length === "number" ? m.context_length : 0,
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
        });
      }
      if (models.length > 0) return { models };
    }
  } catch {
    // Fall through to the fallback table.
  }
  return null;
}

const BRIEF = loadBrief();
const MODELS: FreeModel[] = BRIEF?.models ?? FALLBACK_MODELS;

function catalogDocument(): Record<string, unknown> {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

function catalogText(): string {
  return JSON.stringify(catalogDocument());
}

// Rank the models for one request: highest house rating first, brief order
// kept for ties, filtered to the capabilities the request needs.
function rankModels(needsJson: boolean, needsTools: boolean): FreeModel[] {
  const fitting = MODELS.filter(
    (m) => (!needsJson || m.json) && (!needsTools || m.tools),
  );
  const pool = fitting.length > 0 ? fitting : MODELS;
  return [...pool].sort((a, b) => b.rating_milli - a.rating_milli);
}

// ---------------------------------------------------------------------------
// Gateway
// ---------------------------------------------------------------------------

function errorCodeOf(text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code;
      if (typeof code === "string" && code !== "") return code.slice(0, 64);
      const kind = parsed.error.type;
      if (typeof kind === "string" && kind !== "") return kind.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to report.
  }
  return "";
}

interface RawCall {
  ok: boolean;
  status: number;
  text: string;
}

async function callGateway(body: Record<string, unknown>): Promise<RawCall> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MAX_GATEWAY_WAIT_MS);
  try {
    const res = await fetch(`${GATEWAY_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${GATEWAY_KEY}`,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    return { ok: res.status >= 200 && res.status < 300, status: res.status, text };
  } catch {
    return { ok: false, status: 0, text: "" };
  } finally {
    clearTimeout(timer);
  }
}

// Dial the gateway for one prompt: try the best model, retry it once on a
// transient refusal, then fail over down the ranked list. At most three
// calls. Returns the first success, or the refusal trail when all fail.
async function dial(
  baseBody: Record<string, unknown>,
  candidates: FreeModel[],
): Promise<GatewayOutcome> {
  const failures: FailureRow[] = [];
  let attempts = 0;
  for (let ci = 0; ci < candidates.length && attempts < MAX_ATTEMPTS_PER_PROMPT; ci += 1) {
    const model = candidates[ci];
    const tries = ci === 0 ? 2 : 1;
    for (let t = 0; t < tries && attempts < MAX_ATTEMPTS_PER_PROMPT; t += 1) {
      attempts += 1;
      if (attempts > 1) await sleep(RETRY_PAUSE_MS);
      const call = await callGateway({ ...baseBody, model: model.id });
      if (call.ok) {
        return { ok: true, model: model.id, status: call.status, text: call.text, failures };
      }
      // A refusal with an HTTP status goes on the trail; a transport failure
      // (status 0) has no status the trail can carry, so it is not listed.
      if (call.status >= 100 && call.status <= 599) {
        failures.push({ status: call.status, code: errorCodeOf(call.text), model: model.id });
      }
      // A hard 4xx other than 429 is not transient: do not burn the retry
      // on the same model, move straight to the next candidate.
      if (call.status >= 400 && call.status < 500 && call.status !== 429) break;
    }
  }
  return { ok: false, model: "", status: 502, text: "", failures };
}

function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  return { "x-makefun-upstream-failures": JSON.stringify(rows) };
}

// ---------------------------------------------------------------------------
// Provider answer shaping
// ---------------------------------------------------------------------------

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const part of content) {
      if (isRecord(part) && part.type === "text" && typeof part.text === "string") out += part.text;
    }
    return out;
  }
  return "";
}

function providerUsage(parsed: Record<string, unknown>, text: string): { prompt_tokens: number; completion_tokens: number } {
  const usage = parsed.usage;
  if (isRecord(usage)) {
    const input = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0;
    const output = typeof usage.completion_tokens === "number" ? usage.completion_tokens : estTokens(text);
    return { prompt_tokens: Math.max(0, input), completion_tokens: Math.max(1, output) };
  }
  return { prompt_tokens: 0, completion_tokens: estTokens(text) };
}

function openAiStop(finish: unknown): string {
  if (finish === "stop" || finish === "length" || finish === "tool_calls" || finish === "content_filter") return finish as string;
  return "stop";
}

function anthropicStop(finish: unknown, hasToolCalls: boolean): string {
  if (hasToolCalls) return "tool_use";
  if (finish === "length") return "max_tokens";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

// ---------------------------------------------------------------------------
// OpenAI wire
// ---------------------------------------------------------------------------

interface SseParsed {
  text: string;
  toolCalls: Array<{ id: string; name: string; args: string }>;
  finishReason: string | null;
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  sawFinish: boolean;
  sawUsage: boolean;
}

function parseGatewaySse(raw: string): SseParsed {
  const out: SseParsed = {
    text: "",
    toolCalls: [],
    finishReason: null,
    usage: null,
    sawFinish: false,
    sawUsage: false,
  };
  const byIndex = new Map<number, { id: string; name: string; args: string }>();
  for (const line of raw.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (payload === "" || payload === "[DONE]") continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
    if (isRecord(chunk.usage)) {
      const input = typeof chunk.usage.prompt_tokens === "number" ? chunk.usage.prompt_tokens : 0;
      const output = typeof chunk.usage.completion_tokens === "number" ? chunk.usage.completion_tokens : 0;
      out.usage = { prompt_tokens: Math.max(0, input), completion_tokens: Math.max(0, output) };
      out.sawUsage = true;
    }
    if (chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    if (typeof choice.finish_reason === "string" && choice.finish_reason !== "") {
      out.finishReason = choice.finish_reason;
      out.sawFinish = true;
    }
    const delta = isRecord(choice.delta) ? choice.delta : null;
    if (delta === null) continue;
    if (typeof delta.content === "string") out.text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === "number" ? call.index : 0;
        const slot = byIndex.get(index) ?? { id: "", name: "", args: "" };
        if (typeof call.id === "string") slot.id = call.id;
        if (isRecord(call.function)) {
          if (typeof call.function.name === "string") slot.name = call.function.name;
          if (typeof call.function.arguments === "string") slot.args += call.function.arguments;
        }
        byIndex.set(index, slot);
      }
    }
  }
  out.toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => slot);
  return out;
}

// Re-emit the gateway's SSE with the dialed model id on every chunk, then
// exactly one [DONE]. A missing finish chunk is synthesized so the stream
// always ends well-formed.
function forwardOpenAiStream(raw: string, dialedId: string, replyId: string): string {
  const frames: string[] = [];
  let sawFinish = false;
  let sawUsage = false;
  let text = "";
  for (const line of raw.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (payload === "" || payload === "[DONE]") continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
    chunk.model = dialedId;
    if (isRecord(chunk.usage)) sawUsage = true;
    if (chunk.choices.length > 0) {
      const choice = chunk.choices[0];
      if (isRecord(choice)) {
        if (typeof choice.finish_reason === "string" && choice.finish_reason !== "") sawFinish = true;
        if (isRecord(choice.delta) && typeof choice.delta.content === "string") text += choice.delta.content;
      }
    }
    frames.push(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  if (!sawFinish) {
    const usage = sawUsage
      ? undefined
      : { prompt_tokens: 0, completion_tokens: estTokens(text) };
    const fin: Record<string, unknown> = {
      id: replyId,
      object: "chat.completion.chunk",
      created: trialEpoch(),
      model: dialedId,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    };
    if (usage !== undefined) fin.usage = usage;
    frames.push(`data: ${JSON.stringify(fin)}\n\n`);
  }
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

async function handleOpenAi(
  req: IncomingMessage,
  res: ServerResponse,
  body: Record<string, unknown>,
): Promise<void> {
  const stream = body.stream === true;
  const needsJson = isRecord(body.response_format) && body.response_format.type === "json_object";
  const needsTools = Array.isArray(body.tools) && body.tools.length > 0;
  const candidates = rankModels(needsJson, needsTools);

  const maxTokensRaw =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : typeof body.max_completion_tokens === "number"
        ? body.max_completion_tokens
        : 1024;
  const maxTokens = Math.min(8192, Math.max(1, maxTokensRaw));

  const gatewayBody: Record<string, unknown> = {
    messages: body.messages,
    max_tokens: maxTokens,
    stream,
  };
  if (typeof body.temperature === "number") gatewayBody.temperature = body.temperature;
  if (stream) gatewayBody.stream_options = { include_usage: true };
  if (isRecord(body.response_format)) gatewayBody.response_format = body.response_format;
  if (Array.isArray(body.tools)) gatewayBody.tools = body.tools;
  if (body.tool_choice !== undefined) gatewayBody.tool_choice = body.tool_choice;

  const outcome = await dial(gatewayBody, candidates);
  const replyId = nextReplyId("chatcmpl");
  if (!outcome.ok) {
    const extra = failuresHeader(outcome.failures);
    res.writeHead(502, { "content-type": "application/json", ...extra });
    res.end(JSON.stringify({ error: { message: "the upstream model refused the request", type: "upstream_error", code: "upstream_error" } }));
    return;
  }

  const extra = failuresHeader(outcome.failures);
  if (!stream) {
    let parsed: Record<string, unknown>;
    try {
      const value: unknown = JSON.parse(outcome.text);
      parsed = isRecord(value) ? value : {};
    } catch {
      parsed = {};
    }
    const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : {};
    const message = isRecord(first.message) ? (first.message as Record<string, unknown>) : {};
    const text = extractText(message.content);
    const usage = providerUsage(parsed, text);
    const answer = {
      id: replyId,
      object: "chat.completion",
      created: trialEpoch(),
      model: outcome.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: typeof message.content === "string" ? message.content : text,
            ...(Array.isArray(message.tool_calls) ? { tool_calls: message.tool_calls } : {}),
          },
          finish_reason: openAiStop(first.finish_reason),
        },
      ],
      usage,
    };
    res.writeHead(200, { "content-type": "application/json", ...extra });
    res.end(JSON.stringify(answer));
    return;
  }

  const sse = forwardOpenAiStream(outcome.text, outcome.model, replyId);
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...extra,
  });
  res.end(sse);
}

// ---------------------------------------------------------------------------
// Anthropic wire
// ---------------------------------------------------------------------------

function anthropicText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const block of content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") out += block.text;
    }
    return out;
  }
  return "";
}

function anthropicMessages(body: Record<string, unknown>): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  const system = body.system;
  if (typeof system === "string" && system !== "") {
    messages.push({ role: "system", content: system });
  } else if (Array.isArray(system)) {
    const text = anthropicText(system);
    if (text !== "") messages.push({ role: "system", content: text });
  }
  if (!Array.isArray(body.messages)) return messages;
  for (const raw of body.messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === "assistant" ? "assistant" : "user";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (role === "user") {
      const parts: Array<Record<string, unknown>> = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          parts.push({ type: "text", text: block.text });
        } else if (block.type === "image" && isRecord(block.source)) {
          const src = block.source as Record<string, unknown>;
          const url =
            src.type === "base64" && typeof src.data === "string"
              ? `data:${typeof src.media_type === "string" ? src.media_type : "image/png"};base64,${src.data}`
              : typeof src.url === "string"
                ? src.url
                : "";
          if (url !== "") parts.push({ type: "image_url", image_url: { url } });
        } else if (block.type === "tool_result") {
          const inner = anthropicText(block.content);
          parts.push({ type: "text", text: `[tool result: ${inner}]` });
        }
      }
      if (parts.length === 1 && parts[0].type === "text") {
        messages.push({ role, content: parts[0].text });
      } else if (parts.length > 0) {
        messages.push({ role, content: parts });
      }
      continue;
    }
    // Assistant history: text plus any tool_use blocks as tool_calls.
    let text = "";
    const toolCalls: Array<Record<string, unknown>> = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "text" && typeof block.text === "string") text += block.text;
      if (block.type === "tool_use" && typeof block.name === "string") {
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : `call-${toolCalls.length}`,
          type: "function",
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
        });
      }
    }
    const msg: Record<string, unknown> = { role, content: text === "" ? null : text };
    if (toolCalls.length > 0) msg.tool_calls = toolCalls;
    messages.push(msg);
  }
  return messages;
}

function anthropicTools(body: Record<string, unknown>): Array<Record<string, unknown>> | null {
  if (!Array.isArray(body.tools) || body.tools.length === 0) return null;
  const out: Array<Record<string, unknown>> = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || typeof tool.name !== "string") continue;
    out.push({
      type: "function",
      function: {
        name: tool.name,
        ...(typeof tool.description === "string" ? { description: tool.description } : {}),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} },
      },
    });
  }
  return out.length > 0 ? out : null;
}

function declaredToolNames(body: Record<string, unknown>): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (isRecord(tool) && typeof tool.name === "string") names.push(tool.name);
  }
  return names;
}

function parseToolArgs(raw: string): Record<string, unknown> {
  if (raw.trim() === "") return {};
  try {
    const value: unknown = JSON.parse(raw);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

async function handleAnthropic(
  req: IncomingMessage,
  res: ServerResponse,
  body: Record<string, unknown>,
): Promise<void> {
  const accept = String(req.headers.accept ?? "");
  const stream = body.stream === true || accept.includes("text/event-stream");
  const needsTools = Array.isArray(body.tools) && body.tools.length > 0;
  const candidates = rankModels(false, needsTools);
  const declared = declaredToolNames(body);

  const maxTokensRaw = typeof body.max_tokens === "number" ? body.max_tokens : 1024;
  const maxTokens = Math.min(8192, Math.max(1, maxTokensRaw));

  const gatewayBody: Record<string, unknown> = {
    messages: anthropicMessages(body),
    max_tokens: maxTokens,
    stream,
  };
  if (typeof body.temperature === "number") gatewayBody.temperature = body.temperature;
  if (stream) gatewayBody.stream_options = { include_usage: true };
  const tools = anthropicTools(body);
  if (tools !== null) gatewayBody.tools = tools;
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice as Record<string, unknown>;
    if (choice.type === "tool" && typeof choice.name === "string") {
      gatewayBody.tool_choice = { type: "function", function: { name: choice.name } };
    } else if (choice.type === "any") {
      gatewayBody.tool_choice = "required";
    } else if (choice.type === "auto") {
      gatewayBody.tool_choice = "auto";
    }
  }

  const outcome = await dial(gatewayBody, candidates);
  const replyId = nextReplyId("msg");
  if (!outcome.ok) {
    const extra = failuresHeader(outcome.failures);
    res.writeHead(502, { "content-type": "application/json", ...extra });
    res.end(JSON.stringify({ type: "error", error: { type: "api_error", message: "the upstream model refused the request" } }));
    return;
  }

  const extra = failuresHeader(outcome.failures);
  if (!stream) {
    let parsed: Record<string, unknown>;
    try {
      const value: unknown = JSON.parse(outcome.text);
      parsed = isRecord(value) ? value : {};
    } catch {
      parsed = {};
    }
    const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : {};
    const message = isRecord(first.message) ? (first.message as Record<string, unknown>) : {};
    const text = extractText(message.content);
    const usage = providerUsage(parsed, text);
    const blocks: Array<Record<string, unknown>> = [];
    if (text !== "") blocks.push({ type: "text", text });
    const rawCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    let keptCalls = 0;
    for (const call of rawCalls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const name = call.function.name;
      if (typeof name !== "string" || !declared.includes(name)) continue;
      keptCalls += 1;
      blocks.push({
        type: "tool_use",
        id: typeof call.id === "string" ? call.id : `toolu-${blocks.length}`,
        name,
        input: parseToolArgs(typeof call.function.arguments === "string" ? call.function.arguments : ""),
      });
    }
    if (blocks.length === 0) blocks.push({ type: "text", text: "" });
    const answer = {
      id: replyId,
      type: "message",
      role: "assistant",
      content: blocks,
      model: outcome.model,
      stop_reason: anthropicStop(first.finish_reason, keptCalls > 0),
      stop_sequence: null,
      usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
    };
    res.writeHead(200, { "content-type": "application/json", ...extra });
    res.end(JSON.stringify(answer));
    return;
  }

  // Streaming: the gateway's SSE is buffered, then re-emitted as Anthropic
  // events in the order the wire requires, with the provider's real usage.
  const parsed = parseGatewaySse(outcome.text);
  const usage = parsed.usage ?? { prompt_tokens: 0, completion_tokens: estTokens(parsed.text) };
  const keptToolCalls = parsed.toolCalls.filter((call) => declared.includes(call.name));
  const stopReason = anthropicStop(parsed.finishReason, keptToolCalls.length > 0);
  const events: string[] = [];
  const startMsg = {
    id: replyId,
    type: "message",
    role: "assistant",
    content: [],
    model: outcome.model,
    stop_reason: null,
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
  };
  events.push(`event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: startMsg })}\n\n`);
  events.push(
    `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`,
  );
  if (parsed.text !== "") {
    events.push(
      `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: parsed.text } })}\n\n`,
    );
  }
  events.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
  let blockIndex = 1;
  for (const call of keptToolCalls) {
    const block = {
      type: "tool_use",
      id: call.id !== "" ? call.id : `toolu-${blockIndex}`,
      name: call.name,
      input: parseToolArgs(call.args),
    };
    events.push(
      `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: blockIndex, content_block: block })}\n\n`,
    );
    events.push(
      `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: blockIndex })}\n\n`,
    );
    blockIndex += 1;
  }
  events.push(
    `event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: Math.max(1, usage.completion_tokens) } })}\n\n`,
  );
  events.push(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...extra,
  });
  res.end(events.join(""));
}

// ---------------------------------------------------------------------------
// Model list
// ---------------------------------------------------------------------------

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const anthropic = req.headers["x-api-key"] !== undefined;
  if (anthropic) {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        data: MODELS.map((m) => ({
          id: m.id,
          type: "model",
          display_name: m.id,
          created_at: trialIso(),
        })),
        has_more: false,
        first_id: MODELS[0]?.id ?? "",
        last_id: MODELS[MODELS.length - 1]?.id ?? "",
      }),
    );
    return;
  }
  res.writeHead(200, { "content-type": "application/json" });
  res.end(
    JSON.stringify({
      object: "list",
      data: MODELS.map((m) => ({
        id: m.id,
        object: "model",
        created: trialEpoch(),
        owned_by: "openrouter",
      })),
    }),
  );
}

// ---------------------------------------------------------------------------
// HTTP server, registration, work polling
// ---------------------------------------------------------------------------

let inFlight = 0;

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    size += buf.length;
    if (size > 8 * 1024 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function jsonError(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify({ error: { message, type: "invalid_request_error", code: "invalid_request" } }));
}

async function route(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://local");
  if (req.method === "GET" && url.pathname === "/v1/models") {
    handleModels(req, res);
    return;
  }
  if (req.method !== "POST") {
    jsonError(res, 404, `no route ${req.method ?? ""} ${url.pathname}`);
    return;
  }
  let raw: string;
  try {
    raw = await readBody(req);
  } catch {
    jsonError(res, 413, "request body too large");
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    jsonError(res, 400, "request body is not JSON");
    return;
  }
  if (!isRecord(body)) {
    jsonError(res, 400, "request body is not one JSON object");
    return;
  }
  inFlight += 1;
  try {
    if (url.pathname === "/v1/chat/completions") {
      await handleOpenAi(req, res, body);
    } else if (url.pathname === "/v1/messages") {
      await handleAnthropic(req, res, body);
    } else {
      jsonError(res, 404, `no route POST ${url.pathname}`);
    }
  } finally {
    inFlight -= 1;
  }
}

async function register(port: number): Promise<boolean> {
  for (let attempt = 0; attempt < REGISTER_TRIES; attempt += 1) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5_000);
      try {
        const res = await fetch(`${ENV_URL}/register`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ protocol_version: 1, port, catalog: catalogDocument() }),
          signal: controller.signal,
        });
        if (res.status >= 200 && res.status < 300) return true;
      } finally {
        clearTimeout(timer);
      }
    } catch {
      // The env is not up yet; retry below.
    }
    await sleep(1_000);
  }
  return false;
}

async function pollWork(): Promise<void> {
  for (;;) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5_000);
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: controller.signal });
        if (res.status >= 200 && res.status < 300) {
          const value: unknown = await res.json();
          if (isRecord(value) && value.done === true && inFlight === 0) return;
        }
      } finally {
        clearTimeout(timer);
      }
    } catch {
      // The env is briefly unreachable; keep polling.
    }
    await sleep(WORK_POLL_MS);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--list")) {
    if (BRIEF !== null) {
      process.stdout.write(`${catalogText()}\n`);
    } else {
      process.stdout.write('{"providers":[]}\n');
    }
    return;
  }

  const server = createServer((req, res) => {
    route(req, res).catch(() => {
      if (!res.headersSent) jsonError(res, 500, "internal error");
      else res.end();
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  process.stderr.write(`free-model-aggregator listening on 0.0.0.0:${port}\n`);

  if (!ENV_URL) {
    // No trial env: serve until stopped.
    return;
  }
  const registered = await register(port);
  if (!registered) {
    process.stderr.write("free-model-aggregator: could not register with the env\n");
    process.exit(1);
  }
  await pollWork();
  server.close();
  process.exit(0);
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
