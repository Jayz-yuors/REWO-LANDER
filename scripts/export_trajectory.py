import argparse
import json
import os
from pathlib import Path

import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(
        description="Record a LunarLander-v3 episode for the Three.js viewer."
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy", choices=["random", "idle", "main", "model"], default="random")
    parser.add_argument("--model", type=str, default=None,
                        help="Stable-Baselines3 model .zip path, required for --policy model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    model = None
    if args.policy == "model":
        if not args.model:
            parser.error("--model PATH is required when --policy model is selected")
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    env = gym.make("LunarLander-v3")
    episodes = []

    try:
        for episode_index in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_index)
            frames = []
            total_reward = 0.0

            # Record the initial state before taking the first action.
            frames.append({
                "step": 0,
                "observation": [float(v) for v in obs],
                "action": 0,
                "reward": 0.0,
                "total_reward": 0.0,
                "terminated": False,
                "truncated": False,
            })

            for step in range(1, args.max_steps + 1):
                if args.policy == "random":
                    action = int(env.action_space.sample())
                elif args.policy == "idle":
                    action = 0
                elif args.policy == "main":
                    action = 2
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                frames.append({
                    "step": step,
                    "observation": [float(v) for v in obs],
                    "action": action,
                    "reward": float(reward),
                    "total_reward": total_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })

                if terminated or truncated:
                    break

            episodes.append({
                "episode": episode_index + 1,
                "policy": args.policy,
                "seed": args.seed + episode_index,
                "total_reward": total_reward,
                "steps": len(frames) - 1,
                "terminated": frames[-1]["terminated"],
                "truncated": frames[-1]["truncated"],
                "frames": frames,
            })
            print(
                f"Episode {episode_index + 1}: steps={len(frames)-1}, "
                f"return={total_reward:.2f}, "
                f"terminated={frames[-1]['terminated']}, "
                f"truncated={frames[-1]['truncated']}"
            )
    finally:
        env.close()

    out_path = Path(__file__).resolve().parents[1] / "visualizer" / "public" / "data" / "trajectory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "environment": "Gymnasium LunarLander-v3 (2D Box2D)",
        "observation_fields": [
            "x_position", "y_position", "x_velocity", "y_velocity",
            "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"
        ],
        "action_names": ["No thrust", "Left engine", "Main engine", "Right engine"],
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved replay data to: {out_path}")


if __name__ == "__main__":
    main()
