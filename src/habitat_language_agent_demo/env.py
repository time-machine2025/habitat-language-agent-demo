from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

Position = Tuple[int, int]


@dataclass
class HouseObservation:
    instruction: str
    agent: Position
    objects: Dict[str, Position]
    receptacles: Dict[str, Position]
    carrying: Optional[str]


class HouseEnv:
    ACTIONS = ("up", "down", "left", "right", "pick", "place")

    def __init__(
        self,
        instruction: str,
        agent: Position,
        objects: Dict[str, Position],
        receptacles: Dict[str, Position],
        blocked: Iterable[Position],
        width: int = 6,
        height: int = 6,
        max_steps: int = 60,
    ) -> None:
        self.instruction = instruction
        self.agent = agent
        self.objects = dict(objects)
        self.receptacles = dict(receptacles)
        self.blocked = set(blocked)
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.carrying: Optional[str] = None
        self.steps_taken = 0

    def clone(self) -> "HouseEnv":
        return HouseEnv(
            instruction=self.instruction,
            agent=self.agent,
            objects=self.objects,
            receptacles=self.receptacles,
            blocked=self.blocked,
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
        )

    def observation(self) -> HouseObservation:
        return HouseObservation(
            instruction=self.instruction,
            agent=self.agent,
            objects=dict(self.objects),
            receptacles=dict(self.receptacles),
            carrying=self.carrying,
        )

    def object_position(self, name: str) -> Position:
        return self.objects[name]

    def receptacle_position(self, name: str) -> Position:
        return self.receptacles[name]

    def done(self) -> bool:
        return self.success() or self.steps_taken >= self.max_steps

    def success(self) -> bool:
        for object_name, position in self.objects.items():
            for receptacle_name, receptacle_pos in self.receptacles.items():
                if object_name in self.instruction and receptacle_name in self.instruction:
                    return position == receptacle_pos and self.carrying is None
        return False

    def step(self, action: str) -> None:
        if action not in self.ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        if self.done():
            return

        self.steps_taken += 1
        x, y = self.agent
        candidate = self.agent
        if action == "up":
            candidate = (x, max(0, y - 1))
        elif action == "down":
            candidate = (x, min(self.height - 1, y + 1))
        elif action == "left":
            candidate = (max(0, x - 1), y)
        elif action == "right":
            candidate = (min(self.width - 1, x + 1), y)
        elif action == "pick":
            for object_name, position in self.objects.items():
                if position == self.agent and self.carrying is None:
                    self.carrying = object_name
                    break
        elif action == "place" and self.carrying is not None:
            self.objects[self.carrying] = self.agent
            self.carrying = None

        if candidate not in self.blocked:
            self.agent = candidate

        if self.carrying is not None:
            self.objects[self.carrying] = self.agent

    def render_ascii(self) -> str:
        rows = []
        for y in range(self.height):
            tokens = []
            for x in range(self.width):
                pos = (x, y)
                token = "."
                if pos in self.blocked:
                    token = "#"
                for name, obj_pos in self.objects.items():
                    if obj_pos == pos:
                        token = name[0].upper()
                for name, rec_pos in self.receptacles.items():
                    if rec_pos == pos:
                        token = name[0].lower()
                if self.agent == pos:
                    token = "A"
                tokens.append(token)
            rows.append(" ".join(tokens))
        return "\n".join(rows)
