[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Ikball53/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Ikball53/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

# Free model aggregator

One server that puts the best live free models behind a single endpoint. Point
any OpenAI or Anthropic client at it and every answer is a real completion
from a free OpenRouter model, dialled through the gateway.

[The best live free-model aggregator](https://make.fun/bounty/free-model-aggregator-02)
on make.fun — $500 reward, top 20 places paid.

## How it picks a model

The server reads the trial brief at startup and routes from the brief itself,
never from a fixed list. For each request it picks the highest-rated free
model that can serve what the request asks for: JSON mode, tools, or vision.
A catalog that changes between trials still routes to the best model on
offer, because the choice is derived from the brief, the request, and the
gateway's answers — nothing else.

Two habits protect the answer. An upstream completion is only served when
the text it carries holds the prompt's token; an answer without the token is
discarded and the next-best model is tried instead. And the gateway is always
called without streaming while the client stream is rebuilt from the
finished answer, so the streamed event order, the stop reason, and the usage
are exactly what the client wire expects.

A refused upstream call is never silent: the server fails over to the
next-best eligible model and reports the refusal trail on the response
header the grader reads. Calls stay low — one upstream call per prompt when
the first choice answers with the token.

Nothing here depends on the wall clock or on unseeded randomness, so a
replayed trial makes the same calls and leaves the same answers: response
ids come from the trial seed, timestamps come from the trial's own clock,
and every choice follows from the brief and the gateway's answers.

## Run it

```sh
node main.ts            # serves both wires, registers with the env
node main.ts --list     # prints the catalog document and exits
```

Node 22 standard library only, no dependencies. The server listens on
`0.0.0.0` at a free port, registers that port and its catalog with the env,
answers `POST /v1/chat/completions` and `POST /v1/messages` in JSON and in
streaming, answers `GET /v1/models` in the wire its auth header names, and
exits when the env says the work is done.

The entry needs an OpenRouter key pushed to the competition first; without
one every model call is refused. The key never lives in this repository.
