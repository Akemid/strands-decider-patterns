"""Serve the CTF simulator as a local streamable-HTTP MCP server (default http://127.0.0.1:8765/mcp).

    uv run python simulator/serve.py --ticks 60 --seed 0

One server process is one match: restart it to play another. The agent in `examples/ctf_agent` connects to
this URL by default (override with `CTF_MCP_URL`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.engine import SimMatch
from simulator.fake_ctf_mcp import FakeGame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Capture the Flag MCP simulator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ticks", type=int, default=60, help="Tick limit; the match is a tie after it.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the enemies' random walk.")
    parser.add_argument("--deadline-ms", type=int, default=5000, help="Per-tick deadline advertised to agents.")
    return parser.parse_args(argv)


def build_game(args: argparse.Namespace) -> FakeGame:
    game = FakeGame(turns=args.ticks, deadline_ms=args.deadline_ms, engine=SimMatch(max_ticks=args.ticks, seed=args.seed))
    game.server.settings.host = args.host
    game.server.settings.port = args.port
    return game


def main() -> None:
    args = parse_args()
    game = build_game(args)
    print(f"CTF simulator on http://{args.host}:{args.port}/mcp ({args.ticks} ticks, seed {args.seed})")
    game.server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
