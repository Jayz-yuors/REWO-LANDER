
"""
Lunar Lander Policy Evaluation

Save:
src/training/evaluate_policy.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "final_policy.zip"
OUTPUT_DIR = ROOT / "results" / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "LunarLander-v3"

N_EPISODES = 100
BASE_SEED = 10_000


def evaluate():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train_ppo.py first."
        )

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    env = gym.make(ENV_ID)

    episode_results = []

    successful_landings = 0
    total_returns = []

    print("=" * 60)
    print("LUNAR LANDER POLICY EVALUATION")
    print("=" * 60)

    try:
        for episode in range(N_EPISODES):

            obs, info = env.reset(
                seed=BASE_SEED + episode
            )

            terminated = False
            truncated = False

            episode_return = 0.0
            steps = 0

            while not (terminated or truncated):

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(int(action))
                )

                episode_return += float(reward)
                steps += 1

            # LunarLander terminal reward is useful
            # for classifying landing/crash outcomes.
            # A return threshold alone is not a
            # universal success definition.
            final_terminal_reward = float(reward)

            landed = (
                terminated
                and not truncated
                and final_terminal_reward > 0
            )

            if landed:
                successful_landings += 1

            total_returns.append(episode_return)

            episode_results.append({
                "episode": episode + 1,
                "seed": BASE_SEED + episode,
                "episode_return": episode_return,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_terminal_reward": (
                    final_terminal_reward
                ),
                "landed": bool(landed),
                "outcome": (
                    "landed"
                    if landed
                    else (
                        "truncated"
                        if truncated
                        else "not_landed"
                    )
                ),
            })

            print(
                f"Episode {episode + 1:03d} | "
                f"Return: {episode_return:8.2f} | "
                f"Outcome: "
                f"{episode_results[-1]['outcome']}"
            )

    finally:
        env.close()

    success_rate = (
        successful_landings / N_EPISODES
    )

    results = {
        "experiment": "PPO LunarLander Evaluation",
        "environment": ENV_ID,
        "model": str(MODEL_PATH),
        "episodes": N_EPISODES,
        "base_seed": BASE_SEED,
        "successes": successful_landings,
        "success_rate": success_rate,
        "mean_return": float(np.mean(total_returns)),
        "std_return": float(np.std(total_returns)),
        "min_return": float(np.min(total_returns)),
        "max_return": float(np.max(total_returns)),
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "episode_results": episode_results,
    }

    output_path = (
        OUTPUT_DIR / "evaluation_summary.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Successes: {successful_landings}/{N_EPISODES}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean return: {np.mean(total_returns):.2f}")
    print(f"Std return: {np.std(total_returns):.2f}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    evaluate()
