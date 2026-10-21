import json

import pandas as pd

from src.services import circuit_facts as cf


class FakeResponse:
    def __init__(self, description, content):
        self.description = description
        self.content = content
        self.is_complete = True


class FakeErgast:
    """Mimics fastf1.ergast.Ergast for one circuit with races 2021-2025."""

    def get_circuits(self, season, round):
        return pd.DataFrame([{"circuitId": "villeneuve", "circuitName": "Circuit Gilles Villeneuve",
                              "locality": "Montreal", "country": "Canada"}])

    def get_race_results(self, circuit, results_position=None, fastest_rank=None, limit=None):
        seasons = [2021, 2022, 2023, 2024, 2025]
        desc = pd.DataFrame([{"season": s, "round": 9} for s in seasons])
        if results_position == 1:
            names = ["Max", "Max", "Lewis", "Max", "George"]
            content = [pd.DataFrame([{"givenName": n, "familyName": "X", "constructorName": "Team"}])
                       for n in names]
        else:
            times = [75.0, 74.5, 73.2, 72.9, 73.5]   # the 2024 / 2025 laps are "future"
            content = [pd.DataFrame({"givenName": ["D"], "familyName": [str(s)],
                                     "fastestLapTime": pd.to_timedelta([t], unit="s")})
                       for s, t in zip(seasons, times)]
        return FakeResponse(desc, content)


def test_only_races_before_the_replayed_one_are_used():
    facts = cf.fetch_circuit_facts(2024, 9, ergast=FakeErgast())
    assert [w["season"] for w in facts["winners"]] == [2023, 2022, 2021]   # newest first, no spoilers
    assert facts["most_wins"] == {"driver": "Max X", "wins": 2}
    assert facts["fastest_lap"] == {"time_s": 73.2, "driver": "D 2023", "season": 2023}
    assert facts["first_gp"] == 2021


def test_cached_and_offline(tmp_path):
    calls = []

    def fetch(year, rnd):
        calls.append((year, rnd))
        return {"winners": [], "circuit_id": "x"}

    assert cf.load_circuit_facts(2024, 9, str(tmp_path), fetch) == {"winners": [], "circuit_id": "x"}
    assert cf.load_circuit_facts(2024, 9, str(tmp_path), fetch)["circuit_id"] == "x"
    assert len(calls) == 1
    assert json.loads((tmp_path / "2024_09.json").read_text())["circuit_id"] == "x"

    def offline(year, rnd):
        raise OSError("no network")

    assert cf.load_circuit_facts(2023, 5, str(tmp_path), offline) is None
    assert cf.load_circuit_facts(None, 5, str(tmp_path), offline) is None
