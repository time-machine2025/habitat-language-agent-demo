from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from habitat_language_agent_demo.executor import Executor
from habitat_language_agent_demo.planner import RulePlanner
from habitat_language_agent_demo.scenarios import load_scenarios


class DemoTest(unittest.TestCase):
    def test_rule_planner_solves_all_scenarios(self) -> None:
        scenarios = load_scenarios(ROOT / "data" / "scenarios.json")
        planner = RulePlanner()
        executor = Executor()
        results = [
            executor.execute(scenario.build_env(), planner.plan(scenario.build_env().observation()))
            for scenario in scenarios
        ]
        self.assertTrue(all(result["success"] for result in results))


if __name__ == "__main__":
    unittest.main()
