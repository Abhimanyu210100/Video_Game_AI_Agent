"""GenAI-driven action selection helpers."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

ACTION_NAMES: Dict[int, str] = {
    0: "A",
    1: "B",
    2: "START",
    3: "SELECT",
    4: "UP",
    5: "DOWN",
    6: "LEFT",
    7: "RIGHT",
}

ACTION_FROM_NAME = {name: idx for idx, name in ACTION_NAMES.items()}


class GenAIAgent:
    """
    Agent wrapper around a user-provided decision function.

    decision_fn gets (observation, prompt) and must return one of:
    A, B, START, SELECT, UP, DOWN, LEFT, RIGHT
    """

    def __init__(self, decision_fn: Callable[[np.ndarray, str], str]):
        self.decision_fn = decision_fn

    def build_prompt(self) -> str:
        return (
            "You are controlling Pokemon Red. Choose exactly one action from: "
            "A, B, START, SELECT, UP, DOWN, LEFT, RIGHT. "
            "Respond with only the action name."
        )

    def predict(self, observation: np.ndarray) -> int:
        raw_action = self.decision_fn(observation, self.build_prompt())
        action_name = str(raw_action).strip().upper()
        if action_name not in ACTION_FROM_NAME:
            # Safe fallback if model output is noisy.
            return ACTION_FROM_NAME["A"]
        return ACTION_FROM_NAME[action_name]
