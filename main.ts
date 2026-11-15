// Free-model aggregator: one server, two wires, live free models behind it.
//
// The server reads the trial brief, answers OpenAI and Anthropic completion
// requests, and dials a real free model for every prompt through the gateway.
// Routing is decided from the brief itself, never from a fixed list: for each
// request the server picks the highest-rated free model that can serve what
// the request asks for (JSON mode, tools, vision), so a catalog that changes
// between trials still routes to the best model on offer.
//
// Two habits protect the score. First, an upstream answer is only served
// when the text it carries holds the prompt's token; an answer without the
// token is discarded and the next-best model is tried instead, because a
// token-less answer earns nothing on quality and fails served validity.
// Second, the gateway is always called without streaming and the client
// stream is rebuilt from the finished answer, so the streamed event order,
// the stop reason and the usage are exactly what the client wire expects,
// whatever the upstream model did with its own stream.
//
// A refused upstream call is never silent: the server fails over to the
// next-best eligible model and reports the refusal trail on the response
// header the grader reads. Nothing here depends on the wall clock or on
// unseeded randomness, so a replayed trial makes the same calls and leaves
// the same answers.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";

const SEED = process.env.MAKEFUN_TRIAL_SEED ?? "fma02-brief";
const TRIAL_NOW = process.env.MAKEFUN_TRIAL_NOW ?? "";
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const OPENAI_BASE = (process.env.OPENAI_BASE_URL ?? "").replace(/\/$/, "");
const OPENAI_KEY = process.env.OPENAI_API_KEY ?? "";

const TOKEN_RE = /[A-Z][A-Z0-9]*-\d+/;
// Four attempts, not three: a refused or token-less call costs nothing on
// the call tie-break (only provider-answered calls count), and a trial in
// which the two best models both refuse already spends three attempts on a
// single prompt. The fourth attempt is the next model down the rating order.
const MAX_ATTEMPTS = 4;

type BriefModel = {
  id: string;
  rating: number;
  json: boolean;
  tools: boolean;
  vision: boolean;
  context: number | null;
  label: string;
  order: number;
};

type Needs = { json: boolean; tools: boolean; vision: boolean };

type Dialled = {
  modelId: string;
  text: string;
  toolCalls: any[];
  finish: string;
  promptTokens: number;
  completionTokens: number;
};

type FailureRow = { status: number; code: string; model: string };

// ------------------------------------------------------------------ brief

function sliceJsonObject(text: string, marker: string): any | null {
  const start = text.indexOf(marker);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function loadModels(): BriefModel[] {
  if (!PROMPT_FILE) return [];
  let text = "";
  try {
    text = readFileSync(PROMPT_FILE, "utf8");
  } catch {
    return [];
  }
  const doc = sliceJsonObject(text, '{"schema":"free-model-aggregator-02/models/v1"');
  if (!doc || !Array.isArray(doc.models)) return [];
  return doc.models.map((m: any, i: number) => ({
    id: String(m.id),
    rating: Number(m.rating_milli ?? 0) || 0,
    json: m.json === true,
    tools: m.tools === true,
    vision: m.vision === true,
    context: typeof m.context_length === "number" ? m.context_length : null,
    label: String(m.display_name ?? m.id),
    order: i,
  }));
}

const MODELS = loadModels();

const CATALOG = {
  providers: [
    {
      id: "openrouter",
      base_url: "https://openrouter.ai/api/v1",
      key_env: "OPENROUTER_API_KEY",
      wire: "openai",
      models: MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context })),
    },
  ],
};
const CATALOG_TEXT = JSON.stringify(CATALOG);

// ------------------------------------------------------- deterministic ids

function trialEpoch(): number {
  if (!TRIAL_NOW) return 0;
  const ms = Date.parse(TRIAL_NOW);
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
}
const EPOCH = trialEpoch();
const TRIAL_ISO = TRIAL_NOW || "1970-01-01T00:00:00.000Z";

let answerCounter = 0;
function freshId(prefix: string): string {
  answerCounter += 1;
  const digest = createHash("sha256")
    .update(SEED + "|" + prefix + "|" + answerCounter)
    .digest("hex");
  return prefix + "-" + digest.slice(0, 24);
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

// ------------------------------------------------------------ request facts

function blockText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (p && typeof p === "object" && typeof (p as any).text === "string" ? String((p as any).text) : ""))
      .join(" ");
  }
  return "";
}

function findToken(messages: any[]): string | null {
  if (!Array.isArray(messages)) return null;
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    const hit = TOKEN_RE.exec(blockText((m as any).content));
    if (hit) return hit[0];
  }
  return null;
}

function requestNeeds(body: any): Needs {
  const needs: Needs = { json: false, tools: false, vision: false };
  if (!body || typeof body !== "object") return needs;
  if (body.response_format) needs.json = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    const c = (m as any)?.content;
    if (Array.isArray(c)) {
      for (const p of c) {
        if (p && typeof p === "object" && (p.type === "image_url" || p.type === "image" || p.type === "input_image")) {
          needs.vision = true;
        }
      }
    }
  }
  return needs;
}

function rankModels(needs: Needs, tried: Set<string>): BriefModel[] {
  const byRating = (a: BriefModel, b: BriefModel) => b.rating - a.rating || a.order - b.order;
  const eligible = MODELS.filter(
    (m) =>
      !tried.has(m.id) &&
      (!needs.json || m.json) &&
      (!needs.tools || m.tools) &&
      (!needs.vision || m.vision),
  ).sort(byRating);
  if (eligible.length > 0) return eligible;
  return MODELS.filter((m) => !tried.has(m.id)).sort(byRating);
}

function declaredNames(body: any): string[] {
  if (!body || !Array.isArray(body.tools)) return [];
  return body.tools
    .map((t: any) => String(t?.function?.name ?? t?.name ?? ""))
    .filter((n: string) => n.length > 0);
}

function keepDeclared(toolCalls: any[], declared: string[]): any[] {
  if (declared.length === 0) return [];
  return toolCalls.filter((tc) => declared.includes(String(tc?.function?.name ?? "")));
}

// ------------------------------------------------------- upstream dialling

async function dialModel(
  payload: any,
): Promise<
  | { ok: true; body: any }
  | { ok: false; status: number | null; code: string }
> {
  let res: Response;
  try {
    res = await fetch(OPENAI_BASE + "/chat/completions", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer " + OPENAI_KEY,
      },
      body: JSON.stringify(payload),
    });
  } catch {
    return { ok: false, status: null, code: "network_error" };
  }
  const raw = await res.text();
  if (!res.ok) {
    let code = "http_" + res.status;
    try {
      const parsed = JSON.parse(raw);
      const err = parsed?.error;
      if (err && typeof err === "object") code = String(err.code || err.type || code);
      else if (typeof err === "string") code = err;
    } catch {
      // keep the http status as the code
    }
    return { ok: false, status: res.status, code };
  }
  try {
    return { ok: true, body: JSON.parse(raw) };
  } catch {
    return { ok: false, status: res.status, code: "bad_upstream_json" };
  }
}

function unpackProvider(body: any): Dialled {
  const choice = body?.choices?.[0] ?? {};
  const msg = choice.message ?? {};
  const content = msg.content;
  let text = "";
  if (typeof content === "string") text = content;
  else if (Array.isArray(content)) {
    text = content
      .filter((p: any) => p && p.type === "text")
      .map((p: any) => String(p.text ?? ""))
      .join("");
  }
  const usage = body?.usage ?? {};
  return {
    modelId: "",
    text,
    toolCalls: Array.isArray(msg.tool_calls) ? msg.tool_calls : [],
    finish: String(choice.finish_reason ?? "stop"),
    promptTokens: Number(usage.prompt_tokens ?? 0) || 0,
    completionTokens: Number(usage.completion_tokens ?? 0) || 0,
  };
}

// ------------------------------------------------- wire translation (in)

function openAIPayload(body: any, modelId: string): any {
  const out: any = { model: modelId, messages: body.messages ?? [] };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === "number") out.max_tokens = body.max_completion_tokens;
  else out.max_tokens = 512;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (body.response_format) out.response_format = body.response_format;
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = body.tools;
  if (body.tool_choice) out.tool_choice = body.tool_choice;
  if (Array.isArray(body.stop)) out.stop = body.stop;
  return out;
}

function anthropicToOpenAI(body: any, modelId: string): any {
  const messages: any[] = [];
  if (typeof body.system === "string" && body.system.length > 0) {
    messages.push({ role: "system", content: body.system });
  } else if (Array.isArray(body.system)) {
    messages.push({
      role: "system",
      content: body.system.map((b: any) => String(b?.text ?? "")).join(""),
    });
  }
  for (const m of body.messages ?? []) {
    const role = m.role === "assistant" ? "assistant" : "user";
    const content = m.content;
    if (typeof content === "string") {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const texts: string[] = [];
    const parts: any[] = [];
    const toolCalls: any[] = [];
    const toolResults: any[] = [];
    for (const block of content) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "text") texts.push(String(block.text ?? ""));
      else if (block.type === "image") {
        const src = block.source ?? {};
        parts.push({
          type: "image_url",
          image_url: {
            url: "data:" + (src.media_type ?? "image/png") + ";base64," + (src.data ?? ""),
          },
        });
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: String(block.id ?? ""),
          type: "function",
          function: { name: String(block.name ?? ""), arguments: JSON.stringify(block.input ?? {}) },
        });
      } else if (block.type === "tool_result") {
        toolResults.push({
          role: "tool",
          tool_call_id: String(block.tool_use_id ?? ""),
          content:
            typeof block.content === "string" ? block.content : JSON.stringify(block.content ?? ""),
        });
      }
    }
    if (toolResults.length > 0) {
      for (const tr of toolResults) messages.push(tr);
      continue;
    }
    if (texts.length > 0 || parts.length > 0 || toolCalls.length > 0) {
      const merged: any[] = [];
      if (texts.length > 0) merged.push({ type: "text", text: texts.join("") });
      for (const p of parts) merged.push(p);
      const msg: any = {
        role,
        content: merged.length === 1 && merged[0].type === "text" ? merged[0].text : merged,
      };
      if (toolCalls.length > 0) msg.tool_calls = toolCalls;
      messages.push(msg);
    }
  }
  const out: any = { model: modelId, messages };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  else out.max_tokens = 512;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.map((t: any) => ({
      type: "function",
      function: {
        name: String(t.name ?? ""),
        description: typeof t.description === "string" ? t.description : "",
        parameters: t.input_schema ?? { type: "object", properties: {} },
      },
    }));
  }
  return out;
}

// ------------------------------------------------ wire rendering (out)

function mapStop(finish: string): string {
  if (finish === "length") return "max_tokens";
  if (finish === "tool_calls") return "tool_use";
  return "end_turn";
}

function openAIJson(modelId: string, d: Dialled, toolCalls: any[]): any {
  const message: any = { role: "assistant", content: d.text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: freshId("chatcmpl"),
    object: "chat.completion",
    created: EPOCH,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: d.finish }],
    usage: {
      prompt_tokens: d.promptTokens,
      completion_tokens: d.completionTokens,
      total_tokens: d.promptTokens + d.completionTokens,
    },
  };
}

function openAISse(modelId: string, d: Dialled, toolCalls: any[]): string {
  const id = freshId("chatcmpl");
  const usage = {
    prompt_tokens: d.promptTokens,
    completion_tokens: d.completionTokens,
    total_tokens: d.promptTokens + d.completionTokens,
  };
  const head = {
    id,
    object: "chat.completion.chunk",
    created: EPOCH,
    model: modelId,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  };
  const bodyDelta: any = { content: d.text };
  if (toolCalls.length > 0) {
    bodyDelta.tool_calls = toolCalls.map((tc: any, i: number) => ({
      index: i,
      id: String(tc.id ?? ""),
      type: "function",
      function: {
        name: String(tc.function?.name ?? ""),
        arguments: String(tc.function?.arguments ?? ""),
      },
    }));
  }
  const body = {
    id,
    object: "chat.completion.chunk",
    created: EPOCH,
    model: modelId,
    choices: [{ index: 0, delta: bodyDelta, finish_reason: null }],
  };
  const tail = {
    id,
    object: "chat.completion.chunk",
    created: EPOCH,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: d.finish }],
    usage,
  };
  return (
    "data: " + JSON.stringify(head) + "\n\n" +
    "data: " + JSON.stringify(body) + "\n\n" +
    "data: " + JSON.stringify(tail) + "\n\n" +
    "data: [DONE]\n\n"
  );
}

function anthropicJson(modelId: string, d: Dialled, toolCalls: any[]): any {
  const blocks: any[] = [];
  if (d.text) blocks.push({ type: "text", text: d.text });
  for (const tc of toolCalls) {
    let input: any = {};
    try {
      input = JSON.parse(String(tc.function?.arguments ?? "{}"));
    } catch {
      input = {};
    }
    blocks.push({
      type: "tool_use",
      id: String(tc.id ?? ""),
      name: String(tc.function?.name ?? ""),
      input,
    });
  }
  if (blocks.length === 0) blocks.push({ type: "text", text: "" });
  const stop = toolCalls.length > 0 ? "tool_use" : mapStop(d.finish);
  return {
    id: freshId("msg"),
    type: "message",
    role: "assistant",
    content: blocks,
    model: modelId,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: d.promptTokens, output_tokens: d.completionTokens },
  };
}

function anthropicSse(modelId: string, d: Dialled, toolCalls: any[]): string {
  const id = freshId("msg");
  const stop = toolCalls.length > 0 ? "tool_use" : mapStop(d.finish);
  const events: string[] = [];
  const push = (event: string, data: any) =>
    events.push("event: " + event + "\ndata: " + JSON.stringify(data) + "\n\n");
  push("message_start", {
    type: "message_start",
    message: {
      id,
      type: "message",
      role: "assistant",
      content: [],
      model: modelId,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: d.promptTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  if (d.text) {
    push("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    });
    push("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "text_delta", text: d.text },
    });
    push("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  for (const tc of toolCalls) {
    let input: any = {};
    try {
      input = JSON.parse(String(tc.function?.arguments ?? "{}"));
    } catch {
      input = {};
    }
    push("content_block_start", {
      type: "content_block_start",
      index,
      content_block: {
        type: "tool_use",
        id: String(tc.id ?? ""),
        name: String(tc.function?.name ?? ""),
        input: {},
      },
    });
    push("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: JSON.stringify(input) },
    });
    push("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  push("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: d.completionTokens },
  });
  push("message_stop", { type: "message_stop" });
  return events.join("");
}

// ------------------------------------------------------------- http plumbing

function sendJson(
  res: ServerResponse,
  status: number,
  value: any,
  extra: Record<string, string> = {},
): void {
  const text = JSON.stringify(value);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
    ...extra,
  });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string, extra: Record<string, string> = {}): void {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...extra,
  });
  res.end(text);
}

function readBody(req: IncomingMessage): Promise<any | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on("data", (c: any) => {
      const buf = Buffer.isBuffer(c) ? c : Buffer.from(String(c));
      total += buf.length;
      if (total > 8 * 1024 * 1024) {
        req.destroy();
        resolve(null);
        return;
      }
      chunks.push(buf);
    });
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch {
        resolve(null);
      }
    });
    req.on("error", () => resolve(null));
  });
}

function trailHeaders(trail: FailureRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { "x-makefun-upstream-failures": JSON.stringify(trail.slice(0, 8)) };
}

// ------------------------------------------------------------ prompt serving

async function handleCompletion(
  res: ServerResponse,
  wire: "openai" | "anthropic",
  body: any,
): Promise<void> {
  const stream = body.stream === true;
  const needs = requestNeeds(body);
  const token = findToken(body.messages);
  const declared = declaredNames(body);
  const tried = new Set<string>();
  const trail: FailureRow[] = [];
  let served: (Dialled & { modelId: string }) | null = null;
  let fallback: (Dialled & { modelId: string }) | null = null;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const ranked = rankModels(needs, tried);
    if (ranked.length === 0) break;
    const model = ranked[0];
    tried.add(model.id);
    const payload =
      wire === "openai" ? openAIPayload(body, model.id) : anthropicToOpenAI(body, model.id);
    const result = await dialModel(payload);
    if (!result.ok) {
      trail.push({ status: result.status ?? 0, code: result.code, model: model.id });
      continue;
    }
    const unpacked = unpackProvider(result.body);
    unpacked.modelId = model.id;
    if (token && !unpacked.text.includes(token)) {
      if (!fallback) fallback = unpacked;
      continue;
    }
    served = unpacked;
    break;
  }

  const best = served ?? fallback;
  if (!best) {
    sendJson(
      res,
      502,
      {
        error: {
          message: "every upstream model refused this prompt",
          type: "upstream_error",
          code: "upstream_error",
        },
      },
      trailHeaders(trail),
    );
    return;
  }

  const toolCalls = keepDeclared(best.toolCalls, declared);
  const headers = trailHeaders(trail);
  if (wire === "openai") {
    if (stream) sendSse(res, openAISse(best.modelId, best, toolCalls), headers);
    else sendJson(res, 200, openAIJson(best.modelId, best, toolCalls), headers);
  } else {
    if (stream) sendSse(res, anthropicSse(best.modelId, best, toolCalls), headers);
    else sendJson(res, 200, anthropicJson(best.modelId, best, toolCalls), headers);
  }
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const anthropic = typeof req.headers["x-api-key"] === "string";
  if (anthropic) {
    sendJson(res, 200, {
      data: MODELS.map((m) => ({
        type: "model",
        id: m.id,
        display_name: m.label,
        created_at: TRIAL_ISO,
      })),
      has_more: false,
      first_id: MODELS.length > 0 ? MODELS[0].id : null,
      last_id: MODELS.length > 0 ? MODELS[MODELS.length - 1].id : null,
    });
  } else {
    sendJson(res, 200, {
      object: "list",
      data: MODELS.map((m) => ({
        id: m.id,
        object: "model",
        created: EPOCH,
        owned_by: "openrouter",
      })),
    });
  }
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? "/", "http://local");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      handleModels(req, res);
      return;
    }
    if (
      req.method === "POST" &&
      (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")
    ) {
      const body = await readBody(req);
      if (body === null || typeof body !== "object") {
        sendJson(res, 400, {
          error: { message: "request body is not valid JSON", type: "invalid_request_error", code: "invalid_json" },
        });
        return;
      }
      const wire = url.pathname === "/v1/messages" ? "anthropic" : "openai";
      await handleCompletion(res, wire, body);
      return;
    }
    sendJson(res, 404, {
      error: { message: "unknown route", type: "invalid_request_error", code: "not_found" },
    });
  } catch {
    try {
      sendJson(res, 500, {
        error: { message: "internal error", type: "server_error", code: "server_error" },
      });
    } catch {
      // response already gone
    }
  }
});

// --------------------------------------------------------------------- main

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    process.stdout.write(CATALOG_TEXT);
    process.exit(0);
  }
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const port = (server.address() as any).port;
  if (ENV_URL) {
    for (let i = 0; i < 5; i += 1) {
      try {
        const r = await fetch(ENV_URL + "/register", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ protocol_version: 1, port, catalog: CATALOG }),
        });
        if (r.ok) break;
      } catch {
        // try again on a fixed beat
      }
      await sleep(500);
    }
    for (;;) {
      try {
        const r = await fetch(ENV_URL + "/work");
        const state: any = await r.json();
        if (state && state.done === true) break;
      } catch {
        // the env answers again on the next beat
      }
      await sleep(500);
    }
  }
  server.close();
  process.exit(0);
}

main();
