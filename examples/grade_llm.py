#!/usr/bin/env python3
"""Grade a language model's play in a world, exactly.

    export OPENAI_API_KEY=...
    python3 examples/grade_llm.py --model gpt-4o-mini
    python3 examples/grade_llm.py --model my-model --base-url http://localhost:8000/v1 \\
        --world worlds/leduc.hunch --agent bob

Any server that speaks the OpenAI chat-completions protocol works; `--header
NAME=VALUE` adds a header a gateway asks for. The model is asked once about every situation its seat can face (Kuhn poker has six).
Each question holds the rules, taken from the comment at the top of the world's
file, what the agent is shown there and the actions it may take. From the
answers Hunch works out, with no sampling, what the model collects against the
solved policy, how much the best reply to the model would take from it, and
which answers cost most.

The questions do not depend on each other, so they are sent in parallel. To
use a batch API or another client instead, write the questions to a file and
grade a file of answers:

    hunch grade worlds/kuhn.hunch --agent alice --questions > questions.jsonl
    hunch grade worlds/kuhn.hunch --agent alice --answers answers.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hunch                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--world", default=None, help="a .hunch file (default: Kuhn poker)")
ap.add_argument("--agent", default=None, help="the seat the model takes (default: the first)")
ap.add_argument("--model", required=True)
ap.add_argument("--base-url", default="https://api.openai.com/v1")
ap.add_argument("--key-env", default="OPENAI_API_KEY", help="environment variable with the key")
ap.add_argument("--mix", action="store_true",
                help="ask for probabilities over the actions instead of one action")
ap.add_argument("--parallel", type=int, default=8)
ap.add_argument("--header", action="append", default=[], metavar="NAME=VALUE",
                help="an extra request header; {session} in VALUE becomes an id made for this run")
ap.add_argument("--save", help="write the model's raw answers here, one JSON line per question")
args = ap.parse_args()

HEADERS = {"Content-Type": "application/json",
           "Authorization": "Bearer " + os.environ.get(args.key_env, ""),
           # Say what is calling: gateways sort traffic by it, and urllib's own
           # name would hide this script behind every other Python program.
           "User-Agent": f"hunch-grade/{hunch.__version__}"}
SESSION = uuid.uuid4().hex
for pair in args.header:
    name, _, value = pair.partition("=")
    HEADERS[name.strip()] = value.strip().replace("{session}", SESSION)

world = hunch.World(args.world) if args.world else hunch.example("kuhn")
agent = args.agent or world.agents[0]
solved = world.train()
questions = world.questions(agent)
print(f"{world.name}: asking {args.model} about {len(questions)} situations as {agent}",
      file=sys.stderr)

HOW = ("Answer with a JSON object giving a probability for each action you may take, "
       'such as {"pass": 0.7, "bet": 0.3}, and nothing else.' if args.mix
       else "Answer with one action name and nothing else.")


def ask(question: dict) -> str:
    prompt = question["prompt"].rsplit("\n", 1)[0] + "\n" + HOW
    body = json.dumps({"model": args.model,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=body, headers=HEADERS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                content = json.load(reply)["choices"][0]["message"].get("content")
            if not content:
                sys.exit(f"{args.model} returned no text for question {question['id']}")
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            # Busy or rate limited: wait and ask again. Anything else is ours to fix.
            if exc.code not in (429, 500, 502, 503, 529) or attempt == 4:
                sys.exit(f"{args.base_url} answered HTTP {exc.code}: {detail}")
            time.sleep(float(exc.headers.get("Retry-After") or 2 ** attempt * 2))


with ThreadPoolExecutor(args.parallel) as pool:
    answers = list(pool.map(ask, questions))
if args.save:
    Path(args.save).write_text("".join(
        json.dumps({"id": q["id"], "sees": q.get("sees"), "answer": a}) + "\n"
        for q, a in zip(questions, answers)))

# An answer may be an action name, JSON probabilities, or a sentence that names
# one legal action; anything else stops here with the text that could not be read.
print(world.grade_answers(answers, agent, against=solved))
