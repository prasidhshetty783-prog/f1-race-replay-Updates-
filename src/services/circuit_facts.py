"""
Circuit history for the 3D Track Detail panel.

Past winners and the fastest race lap on record at the circuit, from the
Jolpica-F1 API (the community successor to Ergast) through FastF1's Ergast
interface. Only races BEFORE the one being replayed are used, so the panel
never spoils the result.

Results are cached in ``<computed data folder>/circuit_facts`` and refreshed
after 30 days; offline, a stale cache is used if there is one.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Callable, Optional

CACHE_MAX_AGE_S = 30 * 24 * 3600
SOURCE = "Jolpica-F1 (Ergast) via FastF1"


def default_cache_dir() -> str:
    try:
        from src.lib.settings import get_settings
        base = get_settings().computed_data_location or "computed_data"
    except Exception:
        base = "computed_data"
    path = os.path.join(base, "circuit_facts")
    os.makedirs(path, exist_ok=True)
    return path


def _all_pages(response):
    """Yield (description_row, content_frame) across every result page."""
    while True:
        for i, (_, row) in enumerate(response.description.iterrows()):
            yield row, response.content[i]
        if getattr(response, "is_complete", True):
            break
        try:
            response = response.get_next_result_page()
        except ValueError:  # no page after this one
            break


def _driver_name(result) -> str:
    return f"{result.get('givenName', '')} {result.get('familyName', '')}".strip()


def _seconds(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        s = value.total_seconds()
        return float(s) if s == s else None  # NaT -> None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_circuit_facts(year: int, round_number: int, ergast=None) -> dict:
    """
    Download circuit history for the race at ``year`` / ``round_number``.

    ``ergast`` is a ``fastf1.ergast.Ergast`` (injectable for tests).
    """
    if ergast is None:
        from fastf1.ergast import Ergast
        ergast = Ergast(result_type="pandas", auto_cast=True, limit=100)

    circuits = ergast.get_circuits(season=year, round=round_number)
    if len(circuits) == 0:
        raise RuntimeError(f"no circuit found for {year} round {round_number}")
    circuit = circuits.iloc[0]
    circuit_id = str(circuit["circuitId"])

    def before_this_race(row) -> bool:
        season, rnd = int(row["season"]), int(row["round"])
        return season < year or (season == year and rnd < round_number)

    winners = []
    for row, results in _all_pages(ergast.get_race_results(circuit=circuit_id, results_position=1, limit=100)):
        if before_this_race(row) and len(results):
            first = results.iloc[0]
            winners.append({"season": int(row["season"]), "driver": _driver_name(first),
                            "team": str(first.get("constructorName", ""))})
    winners.sort(key=lambda w: -w["season"])

    fastest = None
    for row, results in _all_pages(ergast.get_race_results(circuit=circuit_id, fastest_rank=1, limit=100)):
        if not before_this_race(row) or not len(results):
            continue
        lap = results.iloc[0]
        t = _seconds(lap.get("fastestLapTime"))
        if t and (fastest is None or t < fastest["time_s"]):
            fastest = {"time_s": t, "driver": _driver_name(lap), "season": int(row["season"])}

    counts = Counter(w["driver"] for w in winners)
    most = counts.most_common(1)
    seasons = [w["season"] for w in winners]
    return {
        "circuit_id": circuit_id,
        "circuit_name": str(circuit.get("circuitName", "")),
        "locality": str(circuit.get("locality", "")),
        "country": str(circuit.get("country", "")),
        "winners": winners,
        "races_before": len(winners),
        "first_gp": min(seasons) if seasons else None,
        "most_wins": {"driver": most[0][0], "wins": most[0][1]} if most else None,
        "fastest_lap": fastest,
        "source": SOURCE,
    }


def load_circuit_facts(year, round_number, cache_dir: Optional[str] = None,
                       fetch: Optional[Callable[[int, int], dict]] = None) -> Optional[dict]:
    """
    Cached :func:`fetch_circuit_facts`. Returns None when there is no data
    and it can't be downloaded (offline, API down).
    """
    try:
        year, round_number = int(year), int(round_number)
    except (TypeError, ValueError):
        return None
    cache_dir = cache_dir or default_cache_dir()
    path = os.path.join(cache_dir, f"{year}_{round_number:02d}.json")
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_MAX_AGE_S
    if not fresh:
        try:
            facts = (fetch or fetch_circuit_facts)(year, round_number)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(facts, f, indent=1)
            return facts
        except Exception as exc:
            print(f"Circuit history unavailable: {exc}")
            if not os.path.exists(path):
                return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
