from __future__ import annotations

from collections import deque
import random
from typing import Dict, Iterable, List, Tuple

from .env import HouseEnv, Position


class Executor:
    def execute(self, env: HouseEnv, plan: Iterable[Dict[str, str]]) -> Dict[str, object]:
        primitive_actions: List[str] = []
        history = [env.render_ascii()]
        for step in plan:
            action = step["action"]
            if action == "move_to_object":
                primitive_actions.extend(self._move_to(env, env.object_position(step["target"]), history))
            elif action == "move_to_receptacle":
                primitive_actions.extend(self._move_to(env, env.receptacle_position(step["target"]), history))
            elif action in {"pick", "place"}:
                env.step(action)
                primitive_actions.append(action)
                history.append(env.render_ascii())
            if env.done():
                break
        return {
            "success": env.success(),
            "steps_taken": env.steps_taken,
            "primitive_actions": primitive_actions,
            "history": history,
        }

    def _move_to(self, env: HouseEnv, target: Position, history: List[str]) -> List[str]:
        path = shortest_path(env.agent, target, env.blocked, env.width, env.height)
        actions: List[str] = []
        for action in path:
            env.step(action)
            actions.append(action)
            history.append(env.render_ascii())
            if env.done():
                break
        return actions


def shortest_path(
    start: Position,
    target: Position,
    blocked: set[Position],
    width: int,
    height: int,
) -> List[str]:
    if start == target:
        return []
    moves = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        for action, (dx, dy) in moves.items():
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in blocked or nxt in visited:
                continue
            if nxt == target:
                return path + [action]
            visited.add(nxt)
            queue.append((nxt, path + [action]))
    raise RuntimeError(f"No path from {start} to {target}")


def random_baseline(env: HouseEnv, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    actions: List[str] = []
    while not env.done():
        action = rng.choice(HouseEnv.ACTIONS)
        env.step(action)
        actions.append(action)
    return {
        "success": env.success(),
        "steps_taken": env.steps_taken,
        "primitive_actions": actions,
    }
