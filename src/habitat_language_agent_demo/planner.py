from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .env import HouseObservation


class RulePlanner:
    name = "rule"

    def plan(self, observation: HouseObservation) -> List[Dict[str, str]]:
        object_name = _find_entity(observation.instruction, observation.objects)
        receptacle_name = _find_entity(observation.instruction, observation.receptacles)
        return [
            {"action": "move_to_object", "target": object_name, "reason": "Navigate to the named object."},
            {"action": "pick", "target": object_name, "reason": "Pick up the object."},
            {
                "action": "move_to_receptacle",
                "target": receptacle_name,
                "reason": "Navigate to the destination receptacle.",
            },
            {"action": "place", "target": receptacle_name, "reason": "Place the carried object."},
        ]


class OpenAIPlanner:
    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def available(self) -> bool:
        return bool(self.api_key)

    def plan(self, observation: HouseObservation) -> List[Dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce structured plans for a small embodied house agent."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": observation.instruction,
                            "objects": observation.objects,
                            "receptacles": observation.receptacles,
                            "schema": {
                                "plan": [
                                    {"action": "move_to_object", "target": "name", "reason": "text"},
                                    {"action": "pick", "target": "name", "reason": "text"},
                                    {"action": "move_to_receptacle", "target": "name", "reason": "text"},
                                    {"action": "place", "target": "name", "reason": "text"},
                                ]
                            },
                        }
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url="https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI request failed: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        text = body["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError("Could not parse planner output")
        return json.loads(match.group(0))["plan"]


class AutoPlanner:
    name = "auto"

    def __init__(self) -> None:
        self.rule = RulePlanner()
        self.llm = OpenAIPlanner()
        self.last_source = "rule"

    def plan(self, observation: HouseObservation) -> List[Dict[str, str]]:
        if self.llm.available():
            try:
                plan = self.llm.plan(observation)
                self.last_source = f"openai:{self.llm.model}"
                return plan
            except Exception:
                self.last_source = "rule_fallback"
        plan = self.rule.plan(observation)
        if self.last_source != "rule_fallback":
            self.last_source = "rule"
        return plan


def build_planner(mode: str) -> object:
    if mode == "rule":
        return RulePlanner()
    if mode == "openai":
        return OpenAIPlanner()
    if mode == "auto":
        return AutoPlanner()
    raise ValueError(f"Unknown planner mode: {mode}")


def _find_entity(instruction: str, options: Dict[str, object]) -> str:
    lowered = instruction.lower()
    for name in options:
        if name.lower() in lowered:
            return name
    raise ValueError(f"Could not ground instruction against options: {sorted(options)}")
