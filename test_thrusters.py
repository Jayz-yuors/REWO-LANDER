"""Record automated fixed-action Gymnasium tests for the Apollo GLB replay viewer.

No Pygame window or keyboard input is used. Gymnasium/Box2D alone computes physics,
observations, rewards, terrain, and episode termination. The output JSON files can be
loaded directly by web/replay3d/index.html using its Load JSON button.
"""
import json
import time
from pathlib import Path

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as ll

ENV_ID = "LunarLander-v3"
BASE_SEED = 42
MAX_STEPS = 1000
BURST_STEPS = 35
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "raw" / "replays"

# Edit this list before running. Each selected test is recorded as a viewer-compatible JSON.
TESTS_TO_RUN = ["no_thrust", "left_only", "main_only", "right_only", "alternating"]

TESTS = {
    "no_thrust": {"label": "No thrust", "kind": "constant", "action": 0},
    "left_only": {"label": "Left orientation engine burst", "kind": "burst", "action": 1},
    "main_only": {"label": "Main engine burst", "kind": "burst", "action": 2},
    "right_only": {"label": "Right orientation engine burst", "kind": "burst", "action": 3},
    "alternating": {"label": "Alternating thruster sequence", "kind": "sequence", "sequence": [1, 0, 3, 0, 2, 0]},
}


def action_for(name, step):
    cfg = TESTS[name]
    if cfg["kind"] == "constant":
        return cfg["action"]
    if cfg["kind"] == "burst":
        return cfg["action"] if step < BURST_STEPS else 0
    seq = cfg["sequence"]
    return seq[(step // 12) % len(seq)]


def terrain_points(env):
    u = env.unwrapped
    polys = getattr(u, "sky_polys", None)
    if polys:
        return [list(map(float, polys[0][0]))] + [list(map(float, p[1])) for p in polys]
    edges = [[tuple(v) for v in f.shape.vertices] for f in u.moon.fixtures][1:]
    return [list(map(float, edges[0][0]))] + [list(map(float, e[1])) for e in edges]


def outcome(terminated, truncated, last_reward, last_obs):
    if truncated and not terminated:
        return "timeout"
    if terminated and last_reward >= 100:
        return "landed_at_rest"
    if terminated:
        return "out_of_bounds" if abs(float(last_obs[0])) >= 1.0 else "crashed"
    return "unfinished"


def run_one(name, index):
    cfg = TESTS[name]
    # Same seed across tests so the initial terrain/starting conditions are comparable.
    seed = BASE_SEED
    env = gym.make(ENV_ID)  # deliberately no render_mode: no Pygame window
    try:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        rec = {
            "terrain": terrain_points(env),
            "constants": {
                "SCALE": ll.SCALE, "VIEWPORT_W": ll.VIEWPORT_W,
                "VIEWPORT_H": ll.VIEWPORT_H, "LEG_DOWN": ll.LEG_DOWN,
                "helipad_x1": float(u.helipad_x1), "helipad_x2": float(u.helipad_x2),
                "helipad_y": float(u.helipad_y),
            },
            "obs": [obs], "actions": [], "rewards": [],
        }
        terminated = truncated = False
        for step in range(MAX_STEPS):
            action = int(action_for(name, step))
            obs, reward, terminated, truncated, _ = env.step(action)
            rec["obs"].append(obs)
            rec["actions"].append(action)
            rec["rewards"].append(float(reward))
            if terminated or truncated:
                break
        rec["outcome"] = outcome(terminated, truncated, rec["rewards"][-1], obs)
        rec["obs"] = [[round(float(v), 7) for v in row] for row in rec["obs"]]
        rec["rewards"] = [round(float(r), 6) for r in rec["rewards"]]
        rec.update(
            schema="lunarlander-replay/1", env_id=ENV_ID, env_kwargs={}, seed=seed,
            fps=env.metadata["render_fps"], gymnasium_version=gym.__version__,
            policy={"type": "diagnostic", "label": f"AUTOMATED THRUSTER TEST — {cfg['label']}",
                    "checkpoint": None, "test": name, "trained_policy": False},
            created=time.strftime("%Y-%m-%dT%H:%M:%S"), steps=len(rec["actions"]),
            episode_return=round(sum(rec["rewards"]), 4),
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"thruster_{name}_seed{seed}.json"
        path.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        print(f"{cfg['label']}: {rec['steps']} steps, return={rec['episode_return']}, "
              f"outcome={rec['outcome']}\n  Saved: {path}")
    finally:
        env.close()


def main():
    unknown = sorted(set(TESTS_TO_RUN) - set(TESTS))
    if unknown:
        raise ValueError(f"Unknown test names in TESTS_TO_RUN: {unknown}")
    print("Automated Gymnasium-only tests; no Pygame window or keyboard controls.")
    for i, name in enumerate(TESTS_TO_RUN):
        run_one(name, i)
    print("\nNext: start the HTTP server and open web/replay3d/index.html.")
    print("In the viewer, click Load JSON… and choose a thruster_*.json file.")


if __name__ == "__main__":
    main()
