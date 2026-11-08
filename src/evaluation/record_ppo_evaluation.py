
import json
from pathlib import Path
from datetime import datetime, timezone

import gymnasium as gym
from stable_baselines3 import PPO


# ==========================================
# PATHS
# ==========================================
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "final_policy.zip"

SUMMARY_PATH = (
    ROOT / "results" / "training" / "evaluation_summary.json"
)

TEMPLATE_PATH = (
    ROOT / "results" / "raw" / "replays"
    / "replay_random_seed7.json"
)

OUTPUT_DIR = ROOT / "results" / "raw" / "ppo_evaluation"

N_EPISODES = 100
BASE_SEED = 10000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Replay template missing: {TEMPLATE_PATH}"
        )

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        old_summary = json.load(f)

    model = PPO.load(str(MODEL_PATH), device="cpu")
    env = gym.make("LunarLander-v3")

    successes = 0
    results = []

    try:
        for i in range(N_EPISODES):
            seed = BASE_SEED + i

            obs, info = env.reset(seed=seed)

            observations = [obs.tolist()]
            actions = []
            rewards = []

            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                action_int = int(action.item())

                next_obs, reward, terminated, truncated, info = (
                    env.step(action_int)
                )

                actions.append(action_int)
                rewards.append(float(reward))
                observations.append(next_obs.tolist())

                total_reward += float(reward)
                steps += 1
                obs = next_obs

            # Gymnasium LunarLander terminal reward convention:
            # +100 landing, -100 crash.
            final_reward = float(rewards[-1]) if rewards else 0.0

            landed = (
                terminated
                and not truncated
                and final_reward > 0
            )

            if landed:
                outcome = "landed"
                successes += 1
            elif truncated:
                outcome = "truncated"
            elif terminated and final_reward < 0:
                outcome = "crashed"
            else:
                outcome = "not_landed"

            # Clone the old replay schema.
            # Preserve visualization terrain/constants.
            replay = dict(template)

            replay.update({
                "episode_name": f"ppo_eval_seed{seed}",
                "env_id": "LunarLander-v3",
                "seed": seed,
                "fps": 50,
                "obs": observations,
                "actions": actions,
                "rewards": rewards,
                "steps": steps,
                "total_steps": steps,
                "total_reward": total_reward,
                "outcome": outcome,
                "policy": {
                    "type": "PPO",
                    "label": "Trained PPO policy",
                    "checkpoint": "models/final_policy.zip",
                    "deterministic": True
                },
                "recorded_at": datetime.now(
                    timezone.utc
                ).isoformat()
            })

            filename = f"ppo_eval_seed{seed}.json"
            filepath = OUTPUT_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(replay, f, separators=(",", ":"))

            result = {
                "episode": i + 1,
                "seed": seed,
                "steps": steps,
                "episode_return": total_reward,
                "final_terminal_reward": final_reward,
                "terminated": terminated,
                "truncated": truncated,
                "landed": landed,
                "outcome": outcome,
                "replay_file": filename
            }

            results.append(result)

            print(
                f"Episode {i+1:03d} | "
                f"Seed {seed} | "
                f"Return {total_reward:8.2f} | "
                f"Steps {steps:4d} | "
                f"{outcome.upper()}"
            )

    finally:
        env.close()

    success_rate = successes / N_EPISODES

    manifest = {
        "experiment": "PPO LunarLander trajectory recording",
        "model": str(MODEL_PATH),
        "environment": "LunarLander-v3",
        "episodes": N_EPISODES,
        "base_seed": BASE_SEED,
        "successes": successes,
        "success_rate": success_rate,
        "recorded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "episode_results": results
    }

    manifest_path = OUTPUT_DIR / "ppo_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print("PPO RECORDING COMPLETE")
    print("=" * 50)
    print(f"Episodes recorded: {len(results)}")
    print(f"Reported landing successes: {successes}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
