"""
Run 12+ automated Gymnasium LunarLander-v3 diagnostic episodes and save viewer-compatible JSON replays.
No Pygame window, keyboard controls, or user input during simulation.

Run from your project root:
    python scripts/run_12_tests.py

Gymnasium + Box2D computes the physics. JSON files contain recorded observations/actions/rewards;
the Three.js Apollo GLB viewer visualizes those recorded episodes.
"""
import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as ll

ENV_ID = "LunarLander-v3"
BASE_SEED = 42
MAX_STEPS = 1000

# Each test is a fixed action schedule, not a trained landing controller.
# Gymnasium action mapping: 0=no-op, 1=left orientation engine,
# 2=main engine, 3=right orientation engine.
TESTS = [
    ("no_thrust", "No thrust", lambda step: 0),
    ("left_burst_short", "Left engine short burst", lambda step: 1 if step < 10 else 0),
    ("left_burst_long", "Left engine long burst", lambda step: 1 if step < 35 else 0),
    ("main_burst_short", "Main engine short burst", lambda step: 2 if step < 10 else 0),
    ("main_burst_medium", "Main engine medium burst", lambda step: 2 if step < 35 else 0),
    ("main_burst_long", "Main engine long burst", lambda step: 2 if step < 60 else 0),
    ("right_burst_short", "Right engine short burst", lambda step: 3 if step < 10 else 0),
    ("right_burst_long", "Right engine long burst", lambda step: 3 if step < 35 else 0),
    ("alternate_left_right", "Alternating left/right", lambda step: [1, 0, 3, 0][(step // 10) % 4]),
    ("main_then_left", "Main then left", lambda step: 2 if step < 25 else (1 if step < 40 else 0)),
    ("main_then_right", "Main then right", lambda step: 2 if step < 25 else (3 if step < 40 else 0)),
    ("pulse_all_engines", "Pulsed main/left/right", lambda step: [2, 0, 1, 0, 3, 0][(step // 8) % 6]),
]

def terrain_points(env):
    u = env.unwrapped
    polys = getattr(u, "sky_polys", None)
    if polys:
        return [list(map(float, polys[0][0]))] + [list(map(float, p[1])) for p in polys]
    edges = [[tuple(v) for v in f.shape.vertices] for f in u.moon.fixtures][1:]
    return [list(map(float, edges[0][0]))] + [list(map(float, e[1])) for e in edges]

def classify(terminated, truncated, last_reward, last_obs):
    if truncated and not terminated:
        return "timeout"
    if terminated and last_reward >= 100:
        return "landed_at_rest"
    if terminated:
        return "out_of_bounds" if abs(float(last_obs[0])) >= 1.0 else "crashed"
    return "unfinished"

def run_episode(test, idx, seed, out_dir, wind=False):
    key, label, action_fn = test
    kwargs = {"enable_wind": True} if wind else {}
    env = gym.make(ENV_ID, **kwargs)
    try:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        rec = {
            "terrain": terrain_points(env),
            "constants": {
                "SCALE": ll.SCALE, "VIEWPORT_W": ll.VIEWPORT_W,
                "VIEWPORT_H": ll.VIEWPORT_H, "LEG_DOWN": ll.LEG_DOWN,
                "helipad_x1": float(u.helipad_x1),
                "helipad_x2": float(u.helipad_x2),
                "helipad_y": float(u.helipad_y),
            },
            "obs": [obs.tolist() if hasattr(obs, "tolist") else list(obs)],
            "actions": [], "rewards": [],
        }
        terminated = truncated = False
        step = 0
        while not (terminated or truncated) and step < MAX_STEPS:
            action = int(action_fn(step))
            obs, reward, terminated, truncated, _ = env.step(action)
            rec["obs"].append([round(float(v), 7) for v in obs])
            rec["actions"].append(action)
            rec["rewards"].append(round(float(reward), 6))
            step += 1

        outcome = classify(terminated, truncated, rec["rewards"][-1], obs) if rec["rewards"] else "unfinished"
        episode_return = round(sum(rec["rewards"]), 4)
        rec.update({
            "outcome": outcome,
            "schema": "lunarlander-replay/1",
            "env_id": ENV_ID,
            "env_kwargs": kwargs,
            "seed": seed,
            "fps": env.metadata.get("render_fps", 50),
            "gymnasium_version": gym.__version__,
            "policy": {
                "type": "diagnostic",
                "label": f"FIXED THRUSTER TEST — {label} (not a trained agent)",
                "checkpoint": None,
                "test_key": key,
            },
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "steps": len(rec["actions"]),
            "episode_return": episode_return,
        })
        filename = f"thruster_{key}_seed{seed}{'_wind' if wind else ''}.json"
        path = out_dir / filename
        path.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        return {
            "test_key": key, "label": label, "seed": seed,
            "steps": rec["steps"], "episode_return": episode_return,
            "outcome": outcome, "file": str(path),
        }
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", default="results/raw/replays")
    parser.add_argument("--wind", action="store_true", help="enable Gymnasium wind for every test")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, test in enumerate(TESTS):
        seed = args.seed  # same initial seed across tests for a more controlled comparison
        result = run_episode(test, i, seed, out_dir, args.wind)
        results.append(result)
        print(f"{result['label']}: {result['steps']} steps, "
              f"return={result['episode_return']}, outcome={result['outcome']}")
        print(f"  Saved: {result['file']}")

    summary = {
        "schema": "lunarlander-thruster-test-suite/1",
        "environment": ENV_ID,
        "physics_backend": "Gymnasium + Box2D",
        "visualization": "Three.js viewer replays each individual JSON file; this summary is not a replay.",
        "trained_policy": False,
        "wind_enabled": args.wind,
        "test_count": len(results),
        "results": results,
    }
    summary_path = out_dir / "thruster_test_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCompleted {len(results)} tests.")
    print(f"Summary: {summary_path}")
    print("In the Apollo 3D viewer, click Load JSON and choose any thruster_*.json replay.")

if __name__ == "__main__":
    main()
