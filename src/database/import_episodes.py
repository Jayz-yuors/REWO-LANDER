
import json
import sys
from pathlib import Path

from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from db_config import create_connection


# NEW: Import only the 100 PPO evaluation trajectories.
REPLAY_DIR = ROOT / "results" / "raw" / "ppo_evaluation"


def get_numeric_seed(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def get_fps(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) if value > 0 else 50
    return 50


def import_episode(cur, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    if not isinstance(replay, dict):
        raise ValueError("Replay root must be a JSON object")

    obs = replay.get("obs")
    actions = replay.get("actions")
    rewards = replay.get("rewards")

    if not isinstance(obs, list) or not isinstance(actions, list):
        raise ValueError("Missing or invalid obs/actions")

    if len(obs) == 0:
        raise ValueError("Empty observation trajectory")

    if len(actions) != len(obs) - 1:
        raise ValueError(
            f"Alignment mismatch: obs={len(obs)}, "
            f"actions={len(actions)}"
        )

    if rewards is not None:
        if not isinstance(rewards, list):
            raise ValueError("Invalid rewards")

        if len(rewards) != len(actions):
            raise ValueError(
                f"Reward/action mismatch: rewards={len(rewards)}, "
                f"actions={len(actions)}"
            )

    # Calculate the return from actual recorded rewards.
    total_reward = None
    if rewards is not None:
        if all(
            isinstance(r, (int, float)) and not isinstance(r, bool)
            for r in rewards
        ):
            total_reward = float(sum(rewards))

    episode_name = file_path.stem

    # Store policy label as readable text.
    # Preserve complete policy information in metadata.
    policy_info = replay.get("policy", "unknown")

    if isinstance(policy_info, dict):
        policy_label = str(
            policy_info.get("label", policy_info.get("type", "unknown"))
        )
    else:
        policy_label = str(policy_info)

    # Keep non-trajectory metadata.
    metadata = {
        k: v for k, v in replay.items()
        if k not in {
            "terrain",
            "constants",
            "obs",
            "actions",
            "rewards"
        }
    }

    # PostgreSQL JSONB adaptation via psycopg2.extras.Json.
    cur.execute("""
        INSERT INTO rl_simulation_episodes (
            episode_name,
            env_id,
            seed,
            policy,
            gymnasium_version,
            outcome,
            total_steps,
            total_reward,
            fps,
            terrain,
            constants,
            observations,
            actions,
            rewards,
            metadata
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (episode_name) DO NOTHING;
    """, (
        episode_name,
        str(replay.get("env_id", "LunarLander-v3")),
        get_numeric_seed(replay.get("seed")),
        policy_label,
        str(replay.get("gymnasium_version", "unknown")),
        str(replay.get("outcome", "unknown")),
        len(actions),
        total_reward,
        get_fps(replay.get("fps")),
        Json(replay.get("terrain")),
        Json(replay.get("constants")),
        Json(obs),
        Json(actions),
        Json(rewards),
        Json(metadata)
    ))

    return cur.rowcount


def import_all():
    if not REPLAY_DIR.exists():
        raise FileNotFoundError(
            f"PPO recording directory not found: {REPLAY_DIR}"
        )

    # Only individual PPO trajectory files.
    files = sorted(REPLAY_DIR.glob("ppo_eval_seed*.json"))

    if not files:
        raise FileNotFoundError(
            f"No PPO episode files found in: {REPLAY_DIR}"
        )

    print("=" * 55)
    print("PPO EVALUATION EPISODE IMPORT")
    print("=" * 55)
    print("Source:", REPLAY_DIR)
    print("Files found:", len(files))

    inserted = 0
    skipped = 0
    failed = 0

    conn = create_connection()

    try:
        cur = conn.cursor()

        for file_path in files:
            cur.execute("SAVEPOINT episode_import")

            try:
                rowcount = import_episode(cur, file_path)

                cur.execute("RELEASE SAVEPOINT episode_import")

                if rowcount == 1:
                    inserted += 1
                    print("IMPORTED:", file_path.name)
                else:
                    skipped += 1
                    print("DUPLICATE:", file_path.name)

            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT episode_import")
                cur.execute("RELEASE SAVEPOINT episode_import")

                failed += 1
                print("FAILED:", file_path.name, "|", e)

        conn.commit()

        print("\n========== IMPORT SUMMARY ==========")
        print("Files found:", len(files))
        print("Inserted:", inserted)
        print("Skipped (duplicates):", skipped)
        print("Failed:", failed)

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    import_all()
