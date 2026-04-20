from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .env import HouseEnv


@dataclass(frozen=True)
class Scenario:
    name: str
    instruction: str
    agent: Tuple[int, int]
    objects: Dict[str, Tuple[int, int]]
    receptacles: Dict[str, Tuple[int, int]]
    blocked: Tuple[Tuple[int, int], ...]

    def build_env(self) -> HouseEnv:
        return HouseEnv(
            instruction=self.instruction,
            agent=self.agent,
            objects=self.objects,
            receptacles=self.receptacles,
            blocked=self.blocked,
        )


def load_scenarios(path: str | Path) -> List[Scenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios: List[Scenario] = []
    for item in payload:
        scenarios.append(
            Scenario(
                name=item["name"],
                instruction=item["instruction"],
                agent=tuple(item["agent"]),
                objects={key: tuple(value) for key, value in item["objects"].items()},
                receptacles={key: tuple(value) for key, value in item["receptacles"].items()},
                blocked=tuple(tuple(value) for value in item["blocked"]),
            )
        )
    return scenarios


def scenario_names(scenarios: Iterable[Scenario]) -> List[str]:
    return [scenario.name for scenario in scenarios]
