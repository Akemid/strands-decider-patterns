"""Offline threshold sweep over Decider-mode metrics JSONL (no server, no AWS, no cost).

For each threshold t it reports the share of ticks that would be decided by the Decider alone
(confidence >= t) and the share that would need the LLM, plus the confidence distribution.

With `--llm-only --shadow-decider` files (records carrying `shadow_label`, `shadow_confidence` and
`executed_label`) it also reports, per threshold, how often the Decider's pick matched what the LLM executed.

Usage:
    uv run python scripts/threshold_sweep.py runs/bench-*/decider-seed1.jsonl [--out sweep.md]
        [--min 0.30 --max 0.95 --step 0.05]

Decider-mode files only show coverage, not decision quality: a lower threshold sends fewer ticks to the LLM,
but says nothing about whether the Decider's picks at that confidence are any good.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.ctf_agent.metrics import load_records, percentile  # noqa: E402

NOTE = ("This shows coverage only, not decision quality. It assumes the Decider's confidence is the only "
        "gate and ignores the time budget (a tick may still end on `budget`).")


SHADOW_NOTE = ("agreement with the LLM is a proxy for quality, not ground truth: the LLM can be wrong, and "
               "a disagreement is not necessarily a Decider error.")

def thresholds(low: float = 0.30, high: float = 0.95, step: float = 0.05) -> list[float]:
    count = round((high - low) / step)
    return [round(low + i * step, 4) for i in range(count + 1)]


def sweep(confidences: Sequence[float], values: Sequence[float]) -> list[tuple[float, float, float]]:
    """`(threshold, share decided by the Decider, share needing the LLM)` per threshold."""
    total = len(confidences)
    rows = []
    for t in values:
        share = sum(1 for c in confidences if c >= t) / total if total else 0.0
        rows.append((t, share, (1.0 - share) if total else 0.0))
    return rows


def distribution(confidences: Sequence[float]) -> dict[str, float]:
    values = list(confidences)
    return {"min": min(values), "p25": percentile(values, 25), "p50": percentile(values, 50),
            "p75": percentile(values, 75), "max": max(values)}


def histogram(confidences: Sequence[float], bins: int = 10, width: int = 30) -> list[str]:
    counts = [0] * bins
    for c in confidences:
        counts[min(bins - 1, max(0, int(c * bins)))] += 1
    peak = max(counts) or 1
    return [f"{i / bins:.1f}-{(i + 1) / bins:.1f} | {'#' * round(n / peak * width):<{width}} {n}"
            for i, n in enumerate(counts)]


def load_confidences(paths: Sequence[str | Path]) -> list[float]:
    values: list[float] = []
    for path in paths:
        values += [float(r["confidence"]) for r in load_records(path)
                   if isinstance(r.get("confidence"), int | float)]
    return values


def shadow_pairs(records: Iterable[Mapping[str, Any]]) -> tuple[list[tuple[float, bool]], int]:
    """`(shadow_confidence, shadow_label == executed_label)` per shadow tick, plus how many were excluded.

    Ticks with a null executed label (wait, unmatched, no action) cannot be compared and are excluded;
    ticks without a shadow answer (no shadow fields, or a shadow error) are ignored altogether.
    """
    pairs: list[tuple[float, bool]] = []
    excluded = 0
    for r in records:
        conf = r.get("shadow_confidence")
        if r.get("shadow_label") is None or not isinstance(conf, int | float):
            continue
        if r.get("executed_label") is None:
            excluded += 1
            continue
        pairs.append((float(conf), r["shadow_label"] == r["executed_label"]))
    return pairs, excluded

def shadow_sweep(pairs: Sequence[tuple[float, bool]],
                 values: Sequence[float]) -> list[tuple[float, float, float | None, int]]:
    """`(threshold, covered share, agreement among covered or None, covered count)` per threshold."""
    rows = []
    for t in values:
        covered = [agree for conf, agree in pairs if conf >= t]
        share = len(covered) / len(pairs) if pairs else 0.0
        rows.append((t, share, sum(covered) / len(covered) if covered else None, len(covered)))
    return rows

def shadow_buckets(pairs: Sequence[tuple[float, bool]]) -> list[tuple[float, int, float]]:
    """`(bucket low edge, n, agreement)` for each non-empty 0.1-wide confidence bucket (1.0 joins 0.9-1.0)."""
    buckets: dict[int, list[bool]] = {}
    for conf, agree in pairs:
        buckets.setdefault(min(9, max(0, int(conf * 10 + 1e-9))), []).append(agree)
    return [(i / 10, len(v), sum(v) / len(v)) for i, v in sorted(buckets.items())]

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"

def format_shadow(pairs: Sequence[tuple[float, bool]], excluded: int, values: Sequence[float]) -> list[str]:
    lines = ["", "## Shadow agreement (Decider vs the LLM's executed action)", "", f"> {SHADOW_NOTE}", "",
             f"Compared ticks: {len(pairs)}. Excluded (no executed label): {excluded}.", "",
             "| threshold | covered | agreement among covered | n |", "|---|---|---|---|"]
    lines += [f"| {t:.2f} | {_pct(share)} | {_pct(agree)} | {n} |" for t, share, agree, n in shadow_sweep(pairs, values)]
    lines += ["", "### Agreement by confidence bucket", "", "| confidence | agreement | n |", "|---|---|---|"]
    lines += [f"| {low:.1f}-{low + 0.1:.1f} | {_pct(agree)} | {n} |" for low, n, agree in shadow_buckets(pairs)]
    return lines

def format_report(confidences: Sequence[float], values: Sequence[float], sources: Sequence[str | Path],
                  shadow: tuple[Sequence[tuple[float, bool]], int] | None = None) -> str:
    lines = ["# Threshold sweep", "", f"Sources: {', '.join(str(s) for s in sources)}"]
    if confidences:
        d = distribution(confidences)
        lines += [f"Ticks with a Decider confidence: {len(confidences)}", "", f"> {NOTE}", "",
                  "| threshold | path=decider | needs LLM |", "|---|---|---|"]
        lines += [f"| {t:.2f} | {dec:.0%} | {llm:.0%} |" for t, dec, llm in sweep(confidences, values)]
        lines += ["", "## Confidence distribution", "",
                  "| min | p25 | p50 | p75 | max |", "|---|---|---|---|---|",
                  "| " + " | ".join(f"{d[k]:.2f}" for k in ("min", "p25", "p50", "p75", "max")) + " |",
                  "", "```", *histogram(confidences), "```"]
    if shadow is not None:
        lines += format_shadow(shadow[0], shadow[1], values)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", nargs="+", type=Path, help="One or more metrics JSONL files (decider mode, or llm-only with shadow).")
    p.add_argument("--min", type=float, default=0.30, dest="low")
    p.add_argument("--max", type=float, default=0.95, dest="high")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None, help="Also write the markdown report to this file.")
    args = p.parse_args(argv)
    if args.step <= 0 or args.high < args.low:
        print("error: need --step > 0 and --max >= --min", file=sys.stderr)
        return 2
    records = [r for path in args.metrics for r in load_records(path)]
    confidences = load_confidences(args.metrics)
    pairs, excluded = shadow_pairs(records)
    if not confidences and not pairs:
        print("error: no tick with a numeric confidence or a shadow answer found "
              "(use decider-mode metrics, or --llm-only --shadow-decider metrics)", file=sys.stderr)
        return 2
    shadow = (pairs, excluded) if pairs else None
    report = format_report(confidences, thresholds(args.low, args.high, args.step), args.metrics, shadow)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
