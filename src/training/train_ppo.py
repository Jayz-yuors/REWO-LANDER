
"""
Lunar Lander RL Training
Algorithm: PPO
Environment: Gymnasium LunarLander-v3
Physics: Box2D
Device: CPU

Save:
src/training/train_ppo.py
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "results" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

ENV_ID = "LunarLander-v3"

SEED = 42
TOTAL_TIMESTEPS = 500_000

CHECKPOINT_INTERVAL = 50_000

LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 64
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01

# Optional engine usage penalty per action.
# Start conservatively.
ENGINE_PENALTY = 0.02


# --------------------------------------------------
# REWARD SHAPING
# --------------------------------------------------

class EngineRewardWrapper(gym.Wrapper):
    """
    Preserve Gymnasium reward and add a small
    penalty for engine activation.

    This is an experimental reward configuration.
    It is not a physical fuel model.
    """

    def __init__(self, env, engine_penalty=0.02):
        super().__init__(env)
        self.engine_penalty = engine_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # LunarLander-v3 discrete actions:
        # 0 = no thrust
        # 1 = left orientation engine
        # 2 = main engine
        # 3 = right orientation engine

        engine_active = int(action) != 0

        penalty = (
            self.engine_penalty
            if engine_active
            else 0.0
        )

        shaped_reward = float(reward - penalty)

        # Preserve original environment reward.
        info["base_reward"] = float(reward)
        info["engine_penalty"] = float(penalty)
        info["shaped_reward"] = shaped_reward

        return (
            obs,
            shaped_reward,
            terminated,
            truncated,
            info,
        )


# --------------------------------------------------
# ENVIRONMENT FACTORY
# --------------------------------------------------

def make_env():
    def _init():
        env = gym.make(ENV_ID)

        env = EngineRewardWrapper(
            env,
            engine_penalty=ENGINE_PENALTY,
        )

        env = Monitor(env)

        return env

    return _init


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():

    print("=" * 60)
    print("LUNAR LANDER PPO TRAINING")
    print("=" * 60)

    print(f"Environment: {ENV_ID}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Seed: {SEED}")
    print(f"Device: CPU")
    print(f"Engine penalty: {ENGINE_PENALTY}")

    env = DummyVecEnv([make_env()])

    env.seed(SEED)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            CHECKPOINT_INTERVAL // env.num_envs,
            1,
        ),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_lunarlander",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        seed=SEED,
        device="cpu",
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="PPO_LunarLander",
            progress_bar=False,
        )

        final_path = MODEL_DIR / "final_policy"

        model.save(str(final_path))

        print("\nTraining completed.")
        print(f"Final policy: {final_path}.zip")

    except KeyboardInterrupt:
        interrupted_path = MODEL_DIR / "interrupted_policy"
        model.save(str(interrupted_path))

        print("\nTraining interrupted.")
        print(f"Saved current policy: {interrupted_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
