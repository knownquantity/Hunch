#!/usr/bin/env python3
"""The smallest agent `hunch grade --cmd` can talk to: it reads one JSON
question per line and prints one action per line. This one bets whenever it may.

    hunch train worlds/kuhn.hunch
    hunch grade worlds/kuhn.hunch --agent alice --cmd 'python3 examples/always_bet.py'
"""
import json
import sys

for line in sys.stdin:
    q = json.loads(line)        # agent, obs, sees, legal, prompt
    print("bet" if "bet" in q["legal"] else q["legal"][0], flush=True)
