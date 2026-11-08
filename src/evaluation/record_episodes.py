"""Record real Gymnasium LunarLander-v3 episodes for the 3D replay viewer.

Gymnasium stays the only physics/RL engine: this script only steps the env and saves what
it returned (observations, actions, rewards, terrain, outcome). Nothing is simulated here.

Usage (from repo root):
  python src/evaluation/record_episodes.py                      # newest models/**/*.zip, else random policy
  python src/evaluation/record_episodes.py --model models/ppo_baseline.zip --seed 7
  python src/evaluation/record_episodes.py --random             # force the labelled random-policy test
  python src/evaluation/record_episodes.py --wind --wind-power 8 --turbulence-power 1.0
"""
import argparse, glob, json, os, time
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as ll

ENV_ID = "LunarLander-v3"


def find_checkpoint(path):
    if path:
        return path if os.path.isfile(path) else None
    zips = glob.glob("models/**/*.zip", recursive=True)
    return max(zips, key=os.path.getmtime) if zips else None


def terrain_points(env):
    """Terrain polyline exactly as built by Gymnasium for this seed (Box2D units)."""
    u = env.unwrapped
    polys = getattr(u, "sky_polys", None)
    if polys:
        return [list(map(float, polys[0][0]))] + [list(map(float, p[1])) for p in polys]
    edges = [[tuple(v) for v in f.shape.vertices] for f in u.moon.fixtures][1:]  # [0] is the base line
    return [list(map(float, edges[0][0]))] + [list(map(float, e[1])) for e in edges]


def classify(terminated, truncated, last_reward, last_obs):
    if truncated and not terminated:
        return "timeout"
    if terminated and last_reward >= 100:
        return "landed_at_rest"  # Gymnasium's +100 terminal reward (lander came to rest)
    if terminated:
        return "out_of_bounds" if abs(last_obs[0]) >= 1.0 else "crashed"
    return "unfinished"


def record(env, seed, act_fn):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    rec = dict(terrain=terrain_points(env),
               constants=dict(SCALE=ll.SCALE, VIEWPORT_W=ll.VIEWPORT_W, VIEWPORT_H=ll.VIEWPORT_H,
                              LEG_DOWN=ll.LEG_DOWN, helipad_x1=float(u.helipad_x1),
                              helipad_x2=float(u.helipad_x2), helipad_y=float(u.helipad_y)),
               obs=[obs], actions=[], rewards=[])
    terminated = truncated = False
    while not (terminated or truncated):
        a = act_fn(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        rec["obs"].append(obs)
        rec["actions"].append(int(a))
        rec["rewards"].append(float(r))
    rec["outcome"] = classify(terminated, truncated, rec["rewards"][-1], obs)
    rec["obs"] = [[round(float(v), 7) for v in o] for o in rec["obs"]]
    rec["rewards"] = [round(r, 6) for r in rec["rewards"]]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="SB3 PPO .zip (default: newest models/**/*.zip)")
    ap.add_argument("--random", action="store_true", help="force random-policy visualization test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--stochastic", action="store_true", help="sample PPO actions (default: deterministic)")
    ap.add_argument("--wind", action="store_true")
    ap.add_argument("--wind-power", type=float, default=15.0)
    ap.add_argument("--turbulence-power", type=float, default=1.5)
    ap.add_argument("--out", default="results/raw/replays")
    args = ap.parse_args()

    kw = dict(enable_wind=True, wind_power=args.wind_power, turbulence_power=args.turbulence_power) if args.wind else {}
    env = gym.make(ENV_ID, **kw)  # no render_mode: identical physics, no particles needed
    ckpt = None if args.random else find_checkpoint(args.model)
    if args.model and not ckpt and not args.random:
        raise SystemExit(f"Checkpoint not found: {args.model}")

    if ckpt:
        from stable_baselines3 import PPO
        model = PPO.load(ckpt, device="cpu")
        act = lambda o: int(model.predict(o, deterministic=not args.stochastic)[0])
        policy = dict(type="ppo", label=f"PPO checkpoint: {os.path.basename(ckpt)}", checkpoint=ckpt,
                      deterministic=not args.stochastic)
    else:
        env.action_space.seed(args.seed)
        act = lambda o: int(env.action_space.sample())
        policy = dict(type="random",
                      label="RANDOM POLICY — visualization test only (not a trained agent)", checkpoint=None)
        print("No PPO checkpoint used -> recording a labelled random-policy episode.")

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.episodes):
        seed = args.seed + i
        rec = record(env, seed, act)
        rec.update(schema="lunarlander-replay/1", env_id=ENV_ID, env_kwargs=kw, seed=seed,
                   fps=env.metadata["render_fps"], gymnasium_version=gym.__version__, policy=policy,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"), steps=len(rec["actions"]),
                   episode_return=round(sum(rec["rewards"]), 4))
        name = f"replay_{policy['type']}_seed{seed}{'_wind' if kw else ''}.json"
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            json.dump(rec, f, separators=(",", ":"))
        print(f"{path}: {rec['steps']} steps, outcome={rec['outcome']}, return={rec['episode_return']}")
    env.close()
    print("View: python -m http.server 8000  ->  http://localhost:8000/web/replay3d/index.html")


if __name__ == "__main__":
    main()
