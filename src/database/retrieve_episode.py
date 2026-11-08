
import json
import argparse
from pathlib import Path

from db_config import create_connection


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "web" / "replay3d" / "replays"


def retrieve_episode(episode_id):
    conn = create_connection()

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT
                episode_id,
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
            FROM rl_simulation_episodes
            WHERE episode_id = %s;
        """, (episode_id,))

        row = cur.fetchone()

        if row is None:
            raise ValueError(
                f"Episode ID {episode_id} not found."
            )

        (
            db_id,
            episode_name,
            env_id,
            seed,
            policy,
            gym_version,
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
        ) = row

        # Reconstruct replay structure
        replay = dict(metadata or {})

        replay.update({
            "episode_id": db_id,
            "episode_name": episode_name,
            "env_id": env_id,
            "seed": seed,
            "policy": policy,
            "gymnasium_version": gym_version,
            "outcome": outcome,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "fps": fps,
            "terrain": terrain,
            "constants": constants,
            "obs": observations,
            "actions": actions,
            "rewards": rewards
        })

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = OUTPUT_DIR / "latest.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(replay, f, indent=2)

        print("\nEpisode retrieved successfully.")
        print("Database ID:", db_id)
        print("Name:", episode_name)
        print("Outcome:", outcome)
        print("Steps:", total_steps)
        print("Return:", total_reward)
        print("Replay file:", output_file)

        print("\nStart your HTTP server from the project root:")
        print("python -m http.server 8000")

        print("\nThen open this URL:")
        print(
            "http://localhost:8000/web/replay3d/"
            "index.html?replay=replays/latest.json"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Database episode_id"
    )

    args = parser.parse_args()

    retrieve_episode(args.id)
