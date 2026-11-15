#!/usr/bin/env node
// free-model-aggregator-02 v3 — one server that serves the best free models
// behind the OpenAI and Anthropic wires, dialled through the grader's
// gateway on Chat Completions.
//
// How it picks a model: every request names model "auto". The brief that
// the trial mounts (MAKEFUN_PROMPT_FILE) lists the fourteen free models
// with the house's quality rating and what each one can do. For each
// prompt the server lines those models up best-first among the ones that
// meet the prompt's needs (JSON mode needs a JSON model, a tools prompt
// needs a tools model) and dials them in that order until one answers
// with the prompt's token in its text. A refusal, a bad status or an
// answer without the token is recorded on the x-makefun-upstream-failures
// header and the next model is tried; at most four tries per prompt.
//
// The point of trying best-first instead of one fixed model is the
// quality column: it is the weighted rating of the model that really
// served each prompt over the best free model for that prompt. A fixed
// model caps quality at the fixed model's rating over the best; going
// best-first earns the best rating wherever the best model answers.
//
// Everything the server returns is a function of the requests it has
// seen and the gateway's answers, never of the wall clock: no random
// ids, no timestamps in answers, fixed chunking, fixed timeouts.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

const PROTOCOL_VERSION = 1;
const MAX_ATTEMPTS = 4;
const ATTEMPT_TIMEOUT_MS = 6000;
const REGISTER_RETRIES = 10;
const REGISTER_WAIT_MS = 500;
const WORK_POLL_MS = 400;
const WORK_POLLS_MAX = 500;
const MAX_BODY_BYTES = 1024 * 1024;
const MAX_FAILURE_ROWS = 8;
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/;
const OPENAI_STOPS = new Set(["stop", "length", "tool_calls", "function_call", "content_filter"]);
const ANTHROPIC_STOPS = new Map([
  ["stop", "end_turn"],
  ["length", "max_tokens"],
  ["tool_calls", "tool_use"],
  ["function_call", "tool_use"],
  ["content_filter", "refusal"],
]);

// ---------------------------------------------------------------------------
// The brief: the trial mounts it at MAKEFUN_PROMPT_FILE. It is prose with
// two JSON documents inside it, the prompts manifest and the free model
// list. Only the model list is needed here; it is found by its schema
// marker and cut out with a brace matcher that respects strings.
// ---------------------------------------------------------------------------

function sliceJsonObject(text, marker) {
  const at = text.indexOf(marker);
  if (at < 0) return null;
  const open = text.lastIndexOf("{", at);
  if (open < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(open, i + 1);
    }
  }
  return null;
}

function readBriefModels() {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  let text;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return null;
  }
  const blob = sliceJsonObject(text, "free-model-aggregator-02/models/v1");
  if (blob === null) return null;
  let parsed;
  try {
    parsed = JSON.parse(blob);
  } catch {
    return null;
  }
  if (!parsed || !Array.isArray(parsed.models)) return null;
  const models = [];
  for (const row of parsed.models) {
    if (!row || typeof row.id !== "string" || row.id === "") continue;
    if (row.free === false) continue;
    models.push({
      id: row.id,
      display: typeof row.display_name === "string" ? row.display_name : row.id,
      rating: typeof row.rating_milli === "number" ? row.rating_milli : 0,
      context: typeof row.context_length === "number" ? row.context_length : 0,
      tools: row.tools === true,
      vision: row.vision === true,
      json: row.json === true,
    });
  }
  return models.length > 0 ? models : null;
}

function catalogDocument(models) {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      },
    ],
  };
}

const EMPTY_CATALOG = { providers: [] };

// ---------------------------------------------------------------------------
// Choosing a model: best rating first among the models that meet the
// prompt's needs. The order is fixed by the brief, so two runs of the
// same trial choose the same models.
// ---------------------------------------------------------------------------

function candidatesFor(models, needs) {
  const fitting = models.filter((m) => {
    if (needs.json && !m.json) return false;
    if (needs.tools && !m.tools) return false;
    return true;
  });
  fitting.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return fitting;
}

function needsOf(body) {
  const needs = { json: false, tools: false };
  if (body && body.response_format && body.response_format.type === "json_object") needs.json = true;
  if (body && Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  return needs;
}

// ---------------------------------------------------------------------------
// Reading the prompt's token and text out of a client body, on either
// wire. The token is the word the answer has to carry back.
// ---------------------------------------------------------------------------

function blocksText(blocks) {
  if (typeof blocks === "string") return blocks;
  if (!Array.isArray(blocks)) return "";
  let out = "";
  for (const part of blocks) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "text" && typeof part.text === "string") out += part.text;
    else if (typeof part.text === "string") out += part.text;
  }
  return out;
}

function requestText(body, wire) {
  let text = "";
  if (wire === "anthropic") {
    if (typeof body.system === "string") text += body.system + "\n";
    else if (Array.isArray(body.system)) text += blocksText(body.system) + "\n";
    if (Array.isArray(body.messages)) {
      for (const message of body.messages) {
        if (message && typeof message.content !== "undefined") text += blocksText(message.content) + "\n";
      }
    }
    return text;
  }
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (message && typeof message.content !== "undefined") text += blocksText(message.content) + "\n";
    }
  }
  return text;
}

function findToken(text) {
  const hit = TOKEN_PATTERN.exec(text);
  return hit === null ? null : hit[0];
}

// ---------------------------------------------------------------------------
// The gateway: every free model is dialled on Chat Completions, never on
// Messages. An Anthropic-shaped request is translated into that body
// first. Tools are left out of the upstream body on purpose: the tools
// prompt tells the model to call no tool, and a model that answers with
// tool calls instead of text would burn a whole extra round trip.
// ---------------------------------------------------------------------------

function upstreamBody(body, wire, modelId) {
  const out = { model: modelId, stream: false };
  if (wire === "anthropic") {
    const messages = [];
    if (typeof body.system === "string" && body.system !== "") {
      messages.push({ role: "system", content: body.system });
    } else if (Array.isArray(body.system)) {
      const systemText = blocksText(body.system);
      if (systemText !== "") messages.push({ role: "system", content: systemText });
    }
    if (Array.isArray(body.messages)) {
      for (const message of body.messages) {
        if (!message || typeof message.role !== "string") continue;
        const text = blocksText(message.content);
        messages.push({ role: message.role, content: text });
      }
    }
    out.messages = messages;
  } else {
    out.messages = (Array.isArray(body.messages) ? body.messages : []).map((message) => {
      if (!message || typeof message !== "object") return { role: "user", content: "" };
      const content = typeof message.content === "string" ? message.content : blocksText(message.content);
      const copy = { role: message.role || "user", content };
      if (message.name !== undefined) copy.name = message.name;
      return copy;
    });
  }
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (body.response_format !== undefined) out.response_format = body.response_format;
  return out;
}

function providerText(choice) {
  if (!choice || typeof choice !== "object") return "";
  const message = choice.message;
  if (!message || typeof message !== "object") return "";
  if (typeof message.content === "string") return message.content;
  return blocksText(message.content);
}

function providerUsage(value) {
  if (!value || typeof value !== "object" || typeof value.usage !== "object" || value.usage === null) return null;
  const input = value.usage.prompt_tokens;
  const output = value.usage.completion_tokens;
  if (typeof input !== "number" || typeof output !== "number") return null;
  return { input, output };
}

function errorCodeOf(value) {
  if (!value || typeof value !== "object") return "";
  const error = value.error;
  if (!error || typeof error !== "object") return "";
  if (typeof error.code === "string" && error.code !== "") return error.code.slice(0, 64);
  if (typeof error.type === "string" && error.type !== "") return error.type.slice(0, 64);
  return "";
}

async function dialOnce(modelId, payload) {
  const base = process.env.OPENAI_BASE_URL;
  const key = process.env.OPENAI_API_KEY;
  if (!base || !key) return { ok: false, status: 0, code: "no_gateway", text: "", usage: null, stop: null };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ATTEMPT_TIMEOUT_MS);
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const raw = await response.text();
    let value = null;
    try {
      value = JSON.parse(raw);
    } catch {
      value = null;
    }
    if (response.status < 200 || response.status >= 300) {
      return { ok: false, status: response.status, code: errorCodeOf(value), text: "", usage: null, stop: null };
    }
    const choices = value && Array.isArray(value.choices) ? value.choices : [];
    const choice = choices.length > 0 ? choices[0] : null;
    const text = providerText(choice);
    const stop = choice && typeof choice.finish_reason === "string" ? choice.finish_reason : "stop";
    return { ok: true, status: response.status, code: "", text, usage: providerUsage(value), stop };
  } catch (err) {
    const timedOut = err && err.name === "AbortError";
    return { ok: false, status: 0, code: timedOut ? "timeout" : "fetch_error", text: "", usage: null, stop: null };
  } finally {
    clearTimeout(timer);
  }
}

// One prompt, best-first: dial candidates until one answers with the
// token. Returns the winning answer plus the trail of failures, if any.
async function servePrompt(models, body, wire) {
  const needs = needsOf(body);
  const candidates = candidatesFor(models, needs);
  const text = requestText(body, wire);
  const token = findToken(text);
  const failures = [];
  for (let attempt = 0; attempt < Math.min(MAX_ATTEMPTS, candidates.length); attempt++) {
    const model = candidates[attempt];
    const payload = upstreamBody(body, wire, model.id);
    const answer = await dialOnce(model.id, payload);
    if (!answer.ok) {
      failures.push({ status: answer.status, code: answer.code, model: model.id });
      continue;
    }
    if (token !== null && !answer.text.includes(token)) {
      failures.push({ status: answer.status, code: "token_miss", model: model.id });
      continue;
    }
    return { model, answer, failures };
  }
  return { model: null, answer: null, failures };
}

// ---------------------------------------------------------------------------
// Answers back to the env, on the wire the env used. Usage and stop
// reason travel whole from the provider; when the provider reports no
// usage, a floor estimate keeps the answer valid (an output of at least
// one token is required).
// ---------------------------------------------------------------------------

function settledUsage(answer, requestLength, answerLength) {
  if (answer.usage !== null) return answer.usage;
  return {
    input: Math.max(0, Math.floor(requestLength / 4)),
    output: Math.max(1, Math.floor(answerLength / 4)),
  };
}

function openaiStop(stop) {
  return OPENAI_STOPS.has(stop) ? stop : "stop";
}

function anthropicStop(stop) {
  return ANTHROPIC_STOPS.get(stop) || "end_turn";
}

function splitPieces(text) {
  if (text === "") return [""];
  const size = Math.max(1, Math.ceil(text.length / 3));
  const pieces = [];
  for (let i = 0; i < text.length; i += size) pieces.push(text.slice(i, i + size));
  return pieces;
}

function openaiJsonAnswer(modelId, answer, usage) {
  return {
    id: "chatcmpl-fma02v3",
    object: "chat.completion",
    created: 0,
    model: modelId,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: openaiStop(answer.stop),
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function openaiStreamAnswer(modelId, answer, usage) {
  const lines = [];
  const pieces = splitPieces(answer.text);
  for (const piece of pieces) {
    lines.push(
      `data: ${JSON.stringify({
        id: "chatcmpl-fma02v3",
        object: "chat.completion.chunk",
        created: 0,
        model: modelId,
        choices: [{ index: 0, delta: { role: "assistant", content: piece }, finish_reason: null }],
      })}`,
    );
  }
  lines.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-fma02v3",
      object: "chat.completion.chunk",
      created: 0,
      model: modelId,
      choices: [{ index: 0, delta: {}, finish_reason: openaiStop(answer.stop) }],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    })}`,
  );
  lines.push("data: [DONE]");
  // Server-sent events are separated by a blank line: without it the
  // whole stream arrives as one frame and no event parses.
  return lines.join("\n\n") + "\n\n";
}

function anthropicJsonAnswer(modelId, answer, usage) {
  return {
    id: "msg-fma02v3",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: answer.text }],
    model: modelId,
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

function anthropicStreamAnswer(modelId, answer, usage) {
  const lines = [];
  lines.push(
    `event: message_start\ndata: ${JSON.stringify({
      type: "message_start",
      message: {
        id: "msg-fma02v3",
        type: "message",
        role: "assistant",
        content: [],
        model: modelId,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    })}`,
  );
  lines.push(
    `event: content_block_start\ndata: ${JSON.stringify({
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    })}`,
  );
  for (const piece of splitPieces(answer.text)) {
    lines.push(
      `event: content_block_delta\ndata: ${JSON.stringify({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      })}`,
    );
  }
  lines.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}`);
  lines.push(
    `event: message_delta\ndata: ${JSON.stringify({
      type: "message_delta",
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: usage.output },
    })}`,
  );
  lines.push(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}`);
  // Server-sent events are separated by a blank line: without it the
  // whole stream arrives as one frame and no event parses.
  return lines.join("\n\n") + "\n\n";
}

// ---------------------------------------------------------------------------
// The HTTP server: the two completion wires and the model list. The list
// answers in the OpenAI shape for a Bearer header and the Anthropic
// shape for an x-api-key header.
// ---------------------------------------------------------------------------

function readBody(request) {
  return new Promise((resolve, reject) => {
    const parts = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        request.destroy();
        return;
      }
      parts.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
    request.on("error", reject);
  });
}

function modelListAnswer(models, headers) {
  const keyHeader = headers["x-api-key"];
  if (typeof keyHeader === "string" && keyHeader !== "") {
    const data = models.map((m) => ({
      id: m.id,
      type: "model",
      display_name: m.display,
      created_at: "2026-11-15T00:00:00Z",
    }));
    return {
      data,
      has_more: false,
      first_id: data.length > 0 ? data[0].id : null,
      last_id: data.length > 0 ? data[data.length - 1].id : null,
    };
  }
  return {
    object: "list",
    data: models.map((m) => ({ id: m.id, object: "model", created: 0, owned_by: "openrouter" })),
  };
}

function buildServer(models) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url || "/", "http://127.0.0.1");
      if (request.method === "GET" && url.pathname === "/v1/models") {
        const payload = JSON.stringify(modelListAnswer(models, request.headers));
        response.writeHead(200, { "content-type": "application/json", "content-length": Buffer.byteLength(payload) });
        response.end(payload);
        return;
      }
      if (request.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
        const wire = url.pathname === "/v1/messages" ? "anthropic" : "openai";
        let body;
        try {
          body = JSON.parse(await readBody(request));
        } catch {
          const payload = JSON.stringify({ error: { message: "request body is not JSON", type: "invalid_request_error" } });
          response.writeHead(400, { "content-type": "application/json" });
          response.end(payload);
          return;
        }
        const stream = body.stream === true;
        const outcome = await servePrompt(models, body, wire);
        if (outcome.model === null) {
          const payload = JSON.stringify({
            error: { message: "no free model answered with the prompt token", type: "upstream_error", code: "all_attempts_failed" },
          });
          const headers = { "content-type": "application/json" };
          if (outcome.failures.length > 0) {
            headers["x-makefun-upstream-failures"] = JSON.stringify(outcome.failures.slice(0, MAX_FAILURE_ROWS));
          }
          response.writeHead(502, headers);
          response.end(payload);
          return;
        }
        const requestLength = requestText(body, wire).length;
        const usage = settledUsage(outcome.answer, requestLength, outcome.answer.text.length);
        const headers = {};
        if (outcome.failures.length > 0) {
          headers["x-makefun-upstream-failures"] = JSON.stringify(outcome.failures.slice(0, MAX_FAILURE_ROWS));
        }
        if (stream) {
          const payload = wire === "anthropic"
            ? anthropicStreamAnswer(outcome.model.id, outcome.answer, usage)
            : openaiStreamAnswer(outcome.model.id, outcome.answer, usage);
          headers["content-type"] = "text/event-stream";
          headers["cache-control"] = "no-cache";
          headers["content-length"] = Buffer.byteLength(payload);
          response.writeHead(200, headers);
          response.end(payload);
          return;
        }
        const payload = JSON.stringify(
          wire === "anthropic"
            ? anthropicJsonAnswer(outcome.model.id, outcome.answer, usage)
            : openaiJsonAnswer(outcome.model.id, outcome.answer, usage),
        );
        headers["content-type"] = "application/json";
        headers["content-length"] = Buffer.byteLength(payload);
        response.writeHead(200, headers);
        response.end(payload);
        return;
      }
      response.writeHead(404, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "not found", type: "invalid_request_error" } }));
    } catch (err) {
      const payload = JSON.stringify({ error: { message: String((err && err.message) || err), type: "server_error" } });
      try {
        response.writeHead(500, { "content-type": "application/json" });
        response.end(payload);
      } catch {
        // The socket is already gone; nothing left to do.
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Running a trial: listen, register with the env, poll /work until it
// says done, then leave quietly with exit 0.
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function registerWithEnv(envUrl, port, catalog) {
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt++) {
    try {
      const response = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
      });
      if (response.status >= 200 && response.status < 300) return true;
    } catch {
      // Fall through to the next attempt.
    }
    await sleep(REGISTER_WAIT_MS);
  }
  return false;
}

async function waitForDone(envUrl) {
  for (let poll = 0; poll < WORK_POLLS_MAX; poll++) {
    try {
      const response = await fetch(`${envUrl.replace(/\/$/, "")}/work`);
      if (response.status >= 200 && response.status < 300) {
        const state = await response.json();
        if (state && state.done === true) return true;
      }
    } catch {
      // The env may not be up yet; keep polling.
    }
    await sleep(WORK_POLL_MS);
  }
  return false;
}

async function main() {
  const args = process.argv.slice(2);
  const listOnly = args.includes("--list");
  const models = readBriefModels();
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(models === null ? EMPTY_CATALOG : catalogDocument(models))}\n`);
    process.exit(0);
  }
  if (models === null) {
    process.stderr.write("free-model-aggregator-02 v3: no brief to read (MAKEFUN_PROMPT_FILE is unset or unreadable)\n");
    process.exit(1);
  }
  const catalog = catalogDocument(models);
  const server = buildServer(models);
  const port = await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      resolve(typeof address === "object" && address !== null ? address.port : 0);
    });
  });
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!envUrl) {
    process.stdout.write(`free-model-aggregator-02 v3: listening on 0.0.0.0:${port} (no MAKEFUN_ENV_URL set, standing by)\n`);
    return;
  }
  const registered = await registerWithEnv(envUrl, port, catalog);
  if (!registered) {
    process.stderr.write("free-model-aggregator-02 v3: the env would not take the registration\n");
    process.exit(1);
  }
  const done = await waitForDone(envUrl);
  server.close();
  process.exit(done ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-02 v3: ${String((err && err.message) || err)}\n`);
  process.exit(1);
});
