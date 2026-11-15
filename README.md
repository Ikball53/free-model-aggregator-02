[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Ikball53/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Ikball53/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

# Free Model Aggregator

One server, two API wires, live free models. Point any OpenAI- or
Anthropic-compatible client at it and every completion is answered by a real
free OpenRouter model, dialled through the grader's gateway.

[The best live free-model aggregator on make.fun](https://make.fun/bounty/free-model-aggregator-02)

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages`
  (Anthropic wire), in JSON and in streaming.
- Serves `GET /v1/models` in the wire the auth header names: `Authorization:
  Bearer` gets the OpenAI list, `x-api-key` the Anthropic one.
- Picks the highest-rated free model that meets each request: JSON mode gets
  the best JSON-capable model, tools get the best tools-capable model, and
  everything else gets the best free model overall.
- Retries a refused model once, then fails over to the next-best model that
  meets the request, and reports recovered refusals in the
  `x-makefun-upstream-failures` header.
- Prints its catalog with `node main.ts --list`, byte for byte as it
  registers it.

## Run

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # serve, register with the env, exit when work is done
```

Node 22, standard library only, nothing to install. The server listens on
`0.0.0.0` at a free port it picks, registers that port and its catalog with
`$MAKEFUN_ENV_URL`, and polls `GET $MAKEFUN_ENV_URL/work` until it says done.

## How it chooses a model

The brief lists fourteen free models with the house's quality rating. The
server ranks them by rating and filters by what the request needs: a
`response_format` of `json_object` needs a JSON-capable model, declared
tools need a tools-capable model. The best model for the request is dialled
first; a 429 or 5xx is retried once on the same model, then the server fails
over down the ranked list. Every choice comes from the brief, the request,
or a gateway answer, so a replay leaves the same state export.

## License

MIT. See [LICENSE](LICENSE).
