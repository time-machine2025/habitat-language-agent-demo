from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from habitat_language_agent_demo.executor import Executor, random_baseline
from habitat_language_agent_demo.planner import build_planner
from habitat_language_agent_demo.scenarios import load_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the habitat language agent demo.")
    parser.add_argument("--planner", choices=["auto", "openai", "rule"], default="auto")
    parser.add_argument("--scenario", default="kitchen_delivery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(ROOT / "data" / "scenarios.json")
    selected = next(item for item in scenarios if item.name == args.scenario)
    env = selected.build_env()
    planner = build_planner(args.planner)
    observation = env.observation()
    plan = planner.plan(observation)

    print("=== Scenario ===")
    print(selected.name)
    print(selected.instruction)
    print()
    print("=== Initial World ===")
    print(env.render_ascii())
    print()
    print("=== Plan ===")
    for index, step in enumerate(plan, start=1):
        print(f"{index}. {step['action']} -> {step['target']} | {step['reason']}")
    print()

    executor = Executor()
    result = executor.execute(env, plan)
    print("=== Final World ===")
    print(env.render_ascii())
    print()
    print("Success:", result["success"])
    print("Steps taken:", result["steps_taken"])
    print()

    successes = []
    steps = []
    random_successes = []
    random_steps = []
    for offset, scenario in enumerate(scenarios):
        env = scenario.build_env()
        rollout = executor.execute(env, planner.plan(env.observation()))
        baseline = random_baseline(scenario.build_env(), seed=offset)
        successes.append(int(rollout["success"]))
        steps.append(rollout["steps_taken"])
        random_successes.append(int(baseline["success"]))
        random_steps.append(baseline["steps_taken"])
    print("=== Benchmark ===")
    print(
        f"Planner success rate: {sum(successes)/len(successes):.2f} | "
        f"avg steps: {statistics.mean(steps):.2f}"
    )
    print(
        f"Random success rate: {sum(random_successes)/len(random_successes):.2f} | "
        f"avg steps: {statistics.mean(random_steps):.2f}"
    )


if __name__ == "__main__":
    main()
