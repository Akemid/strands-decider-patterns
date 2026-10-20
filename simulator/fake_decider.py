"""A scripted stand-in for the Decider server, so the example agent runs end to end without the model.

It is NOT the Decider model: it parses the option texts the CTF agent sends and picks with a few fixed rules
(grab the flag, else the shortest route without a block or a risk). Use it to try the wiring on a laptop;
use `strands-decider serve <checkpoint>` for the real thing.

    uv run python simulator/fake_decider.py --port 8099
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIDENCE = 0.95


def _score(label: str, text: str) -> tuple[int, int]:
    """Lower is better: unblocked and risk-free first, then fewer steps; grab_flag beats everything."""
    if label == "grab_flag":
        return (-1, 0)
    if label == "wait":
        return (9, 0)
    bad = int("BLOCKED" in text) * 2 + int("risk" in text)
    steps = re.search(r"(\d+) steps?", text)
    return (bad, int(steps.group(1)) if steps else 99)


def answer_for(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for key, question in questions.items():
        if question["type"] == "choice":
            options = question["criteria"]
            pick = min(options, key=lambda label: _score(label, options[label]))
            probs = {label: (CONFIDENCE if label == pick else (1 - CONFIDENCE) / max(1, len(options) - 1))
                     for label in options}
            answers[key] = {"type": "choice", "choice": pick, "confidence": CONFIDENCE, "probabilities": probs}
        else:
            answers[key] = {"type": "noul", "noul": 0.0}
    return answers


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # /health
        self._json({"checkpoint": "scripted-stand-in", "base_model": "none", "device": "cpu", "max_length": 0})

    def do_POST(self) -> None:  # /v1/systemone
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self._json({"answers": answer_for(request["questions"]), "usage": {"input_tokens": 0}, "latency_ms": 1.0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted stand-in Decider server.")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    print(f"scripted decider stand-in on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
