"""Run Pokemon Red with a pluggable GenAI decision loop."""

from __future__ import annotations

import argparse
import time

from agent import ACTION_NAMES, GenAIAgent
from emulator import PokemonRedEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pokemon Red with GenAI decisions")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=3_000,
        help="Number of environment steps to run (default: 3,000).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="PyBoy speed multiplier (1-6).",
    )
    parser.add_argument(
        "--fallback-action",
        type=str,
        default="A",
        choices=["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"],
        help="Action used by the default stub if no model is connected.",
    )
    return parser.parse_args()


def default_decision_fn(_observation, _prompt: str, fallback_action: str = "A") -> str:
    """
    Placeholder for your GenAI model call.
    Replace this function body with your API client invocation and parsing logic.
    """
    return fallback_action


def main():
    args = parse_args()
    speed = max(1, min(6, args.speed))

    env = PokemonRedEnv(speed=speed)
    agent = GenAIAgent(
        lambda obs, prompt: default_decision_fn(
            obs, prompt, fallback_action=args.fallback_action
        )
    )

    obs, _ = env.reset()
    total_reward = 0.0

    print(
        f"Running {args.timesteps} steps with GenAI loop "
        f"(fallback action: {args.fallback_action})."
    )
    try:
        for step in range(1, args.timesteps + 1):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                obs, _ = env.reset()
            if step % 100 == 0:
                print(
                    f"step={step} action={ACTION_NAMES[action]} "
                    f"reward_total={total_reward:.2f}"
                )
            # Keep loop stable and visibly watchable at low speed.
            time.sleep(0.001)
    finally:
        env.close()

    print("Run completed successfully.")


if __name__ == "__main__":
    main()
