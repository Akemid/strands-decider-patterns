"""Summarize a decider-mode run against an --llm-only run from their metrics JSONL files.

    uv run python scripts/compare_runs.py runs/decider.jsonl runs/llm-only.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.ctf_agent.metrics import format_comparison, load_records, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decider_run", help="metrics JSONL from a normal (decider mode) run")
    parser.add_argument("llm_run", help="metrics JSONL from a --llm-only run")
    args = parser.parse_args()
    print(format_comparison(summarize(load_records(args.decider_run)), summarize(load_records(args.llm_run))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
