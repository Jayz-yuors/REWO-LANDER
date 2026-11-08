
"""
Record multiple LunarLander episodes into one JSON file.

Physics: Gymnasium LunarLander-v3 / Box2D
Output: results/raw/replays/multi_replay.json

Examples, from the project root:
    python src/evaluation/record_multi_replay.py --random --episodes 12

    python src/evaluation/record_multi_replay.py \
        --model models/your_model.zip --episodes 20
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


# Project root: .../Rocket_RL/
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "raw" / "replays"
OUTPUT_FILE = OUTPUT_DIR / "multi_replay.json"


def classify_outcome(terminated, truncated, total_reward):
    """Classify the final episode outcome."""
    if truncated:
        return "time_limit"
    if terminated and total_reward >= 100:
        return "landed"
    if terminated:
        return "crashed"
    return "incomplete"


def get_terrain(env):
    """Extract the terrain representation used by Gymnasium."""
    try:
        terrain = env.unwrapped
        chunks = terrain.sky_polys
        return [
            [[float(x), float(y)] for x, y in polygon]
            for polygon in chunks
        ]
    except (AttributeError, TypeError):
        return []


def record_episode(env, episode_number, seed, model=None):
    obs, info = env.reset(seed=seed)

    observations = [np.asarray(obs, dtype=float).tolist()]
    actions = []
    rewards = []

    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
        else:
            # Random policy is for testing the recorder/viewer,
            # NOT a trained landing controller.
            action = int(env.action_space.sample())

        next_obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        observations.append(np.asarray(next_obs, dtype=float).tolist())

        total_reward += float(reward)
        step_count += 1
        obs = next_obs

    return {
        "episode_number": episode_number,
        "seed": seed,
        "terrain": get_terrain(env),
        "constants": {
            "gravity": -10.0,
            "enable_wind": False,
            "continuous": False,
        },
        "obs": observations,
        "actions": actions,
        "rewards": rewards,
        "outcome": classify_outcome(
            terminated, truncated, total_reward
        ),
        "env_id": "LunarLander-v3",
        "fps": 50,
        "steps": step_count,
        "episode_return": total_reward,
        "policy": "PPO" if model is not None else "random",
        "gymnasium_version": gym.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--random",
        action="store_true",
        help="Record episodes using random actions for testing",
    )
    group.add_argument(
        "--model",
        type=str,
        help="Path to a trained Stable-Baselines3 PPO .zip model",
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    model = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path

        if not model_path.exists():
            sys.exit(f"Model file not found: {model_path}")

        model = PPO.load(str(model_path), device="cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    episodes = []

    # No render_mode: do not open the Gymnasium Pygame window.
    env = gym.make("LunarLander-v3")

    try:
        for index in range(args.episodes):
            episode_seed = args.seed + index

            episode = record_episode(
                env=env,
                episode_number=index + 1,
                seed=episode_seed,
                model=model,
            )
            episodes.append(episode)

            print(
                f"Episode {index + 1}/{args.episodes} | "
                f"Outcome: {episode['outcome']} | "
                f"Steps: {episode['steps']} | "
                f"Return: {episode['episode_return']:.2f}"
            )
    finally:
        env.close()

    output = {
        "schema": "lunarlander-multi-replay/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env_id": "LunarLander-v3",
        "physics_engine": "Gymnasium / Box2D",
        "visualization": "Three.js",
        "episode_count": len(episodes),
        "episodes": episodes,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved combined replay:\n{OUTPUT_FILE}")
    print(f"Episodes recorded: {len(episodes)}")


if __name__ == "__main__":
    main()
