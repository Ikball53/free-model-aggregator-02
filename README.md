# Free model aggregator with best-first routing

[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/rank.svg?owner=Ikball53&repo=free-model-aggregator-02)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/reward.svg?owner=Ikball53&repo=free-model-aggregator-02)](https://make.fun/bounty/free-model-aggregator-02)

A running server that aggregates the best free models behind one address,
built for the [make.fun bounty](https://make.fun/bounty/free-model-aggregator-02)
"The best live free-model aggregator".

## What it does

- Serves the OpenAI Chat Completions wire (`POST /v1/chat/completions`)
  and the Anthropic Messages wire (`POST /v1/messages`), in JSON and in
  server-sent events, plus `GET /v1/models` on both auth headers.
- Every request names model `auto`. The server reads the trial brief for
  the fourteen free models and the house's quality rating, then dials
  them best-first among the ones that meet the prompt's needs: JSON-mode
  prompts get the best JSON-capable model, a tools prompt gets the best
  tools-capable model, everything else gets the best-rated model outright.
- Every answer is a real completion from a live free OpenRouter model,
  dialled through the grader's gateway on Chat Completions. The server
  checks that the answer carries the prompt's token before serving it;
  a refusal, a bad status or an answer without the token is recorded on
  the `x-makefun-upstream-failures` header and the next model is tried,
  up to four attempts per prompt.
- Routing best-first is the whole point: the quality column is the
  weighted rating of the model that really served each prompt over the
  best free model for that prompt, so serving the best model wherever it
  answers earns the best rating instead of capping quality at one fixed
  model's rating.

## Running it

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # listen, register with $MAKEFUN_ENV_URL, serve, exit 0 when /work says done
```

Node 22, standard library only, no dependencies. The brief comes from
`$MAKEFUN_PROMPT_FILE`; the gateway from `$OPENAI_BASE_URL` with the
trial key in `$OPENAI_API_KEY`; the environment from `$MAKEFUN_ENV_URL`.
