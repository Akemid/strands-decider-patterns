"""Per-tick structured metrics (JSONL) and the decider-vs-LLM comparison summary.

One line per tick: which path decided (decider / llm / budget / wait), how long the Decider and the LLM took,
and token counts when available. `scripts/compare_runs.py` turns two such files into a short report.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self


@dataclass
class TickRecord:
    tick: int
    mode: str  # "decider" or "llm-only"
    path: str  # decider | llm | budget | wait
    decider_ms: float | None = None  # wall time of the Decider request
    llm_ms: float | None = None  # wall time of the LLM fallback run
    label: str | None = None
    confidence: float | None = None
    reason: str | None = None
    llm_timeout: bool = False
    decider_input_tokens: int | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None

    @property
    def decision_ms(self) -> float:
        """Time spent deciding this tick (Decider plus LLM)."""
        return float((self.decider_ms or 0.0) + (self.llm_ms or 0.0))


class MetricsWriter:
    """Appends `TickRecord`s to a JSONL file. With `path=None` it does nothing."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._file = None

    def __enter__(self) -> Self:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: TickRecord) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps({**asdict(record), "decision_ms": record.decision_ms}) + "\n")
        self._file.flush()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class RunSummary:
    ticks: int
    decider_share: float  # fraction of ticks where the Decider's pick ran without the LLM (decider + budget)
    llm_calls: int
    llm_tokens: int
    avg_decision_ms: float | None
    p95_decision_ms: float | None


def _decision_ms(record: Mapping[str, Any]) -> float:
    if record.get("decision_ms") is not None:
        return float(record["decision_ms"])
    return float((record.get("decider_ms") or 0.0) + (record.get("llm_ms") or 0.0))


def summarize(records: list[Mapping[str, Any]]) -> RunSummary:
    ticks = len(records)
    if ticks == 0:
        return RunSummary(0, 0.0, 0, 0, None, None)
    latencies = [_decision_ms(r) for r in records]
    llm_calls = sum(1 for r in records if r.get("llm_ms") is not None or r.get("path") == "llm")
    tokens = sum((r.get("llm_input_tokens") or 0) + (r.get("llm_output_tokens") or 0) for r in records)
    return RunSummary(
        ticks=ticks,
        decider_share=sum(1 for r in records if r.get("path") in ("decider", "budget")) / ticks,
        llm_calls=llm_calls,
        llm_tokens=tokens,
        avg_decision_ms=sum(latencies) / ticks,
        p95_decision_ms=percentile(latencies, 95),
    )


@dataclass(frozen=True)
class Comparison:
    llm_calls_avoided: int
    speedup_avg: float | None


def compare(decider_run: RunSummary, llm_run: RunSummary) -> Comparison:
    avoided = decider_run.ticks - decider_run.llm_calls
    speedup = None
    if decider_run.avg_decision_ms and llm_run.avg_decision_ms:
        speedup = llm_run.avg_decision_ms / decider_run.avg_decision_ms
    return Comparison(avoided, speedup)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def format_comparison(decider_run: RunSummary, llm_run: RunSummary) -> str:
    result = compare(decider_run, llm_run)
    rows = [
        ("ticks", str(decider_run.ticks), str(llm_run.ticks)),
        ("avg decision latency", _ms(decider_run.avg_decision_ms), _ms(llm_run.avg_decision_ms)),
        ("p95 decision latency", _ms(decider_run.p95_decision_ms), _ms(llm_run.p95_decision_ms)),
        ("decisions by Decider", f"{decider_run.decider_share:.0%}", f"{llm_run.decider_share:.0%}"),
        ("LLM calls", str(decider_run.llm_calls), str(llm_run.llm_calls)),
        ("LLM tokens", str(decider_run.llm_tokens), str(llm_run.llm_tokens)),
    ]
    lines = [f"{'metric':<24}{'decider mode':>16}{'llm-only':>16}"]
    lines += [f"{name:<24}{a:>16}{b:>16}" for name, a, b in rows]
    lines.append(f"LLM calls avoided by the Decider: {result.llm_calls_avoided} of {decider_run.ticks} ticks")
    if result.speedup_avg is not None:
        lines.append(f"Average decision latency speedup: {result.speedup_avg:.1f}x")
    return "\n".join(lines)
