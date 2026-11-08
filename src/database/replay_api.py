
"""
Lunar Lander Replay API

Run from project root:
python -m uvicorn src.database.replay_api:app --host 127.0.0.1 --port 8001
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from db_config import create_connection

app = FastAPI(title="Lunar Lander Replay API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Lunar Lander Replay API is running",
        "endpoints": [
            "/api/health",
            "/api/episodes",
            "/api/episodes/{episode_id}/replay",
            "/docs",
        ],
    }


@app.get("/api/health")
def health():
    # Verify database connectivity too
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

        return {
            "status": "ok",
            "database": "connected"
        }
    finally:
        conn.close()


@app.get("/api/episodes")
def list_episodes():
    conn = create_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    episode_id,
                    episode_name,
                    outcome,
                    total_steps,
                    total_reward,
                    seed,
                    policy,
                    created_at
                FROM rl_simulation_episodes
                ORDER BY episode_id ASC
            """)

            rows = cur.fetchall()

        return [
            {
                "episode_id": r[0],
                "episode_name": r[1],
                "outcome": r[2],
                "total_steps": r[3],
                "total_reward": r[4],
                "seed": r[5],
                "policy": r[6],
                "created_at": (
                    r[7].isoformat() if r[7] else None
                )
            }
            for r in rows
        ]

    finally:
        conn.close()


@app.get("/api/episodes/{episode_id}/replay")
def get_replay(episode_id: int):
    conn = create_connection()

    try:
        with conn.cursor() as cur:
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
                WHERE episode_id = %s
            """, (episode_id,))

            row = cur.fetchone()

    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode {episode_id} not found"
        )

    (
        db_id, name, env_id, seed, policy_text,
        gym_version, outcome, total_steps,
        total_reward, fps, terrain, constants,
        obs, actions, rewards, metadata
    ) = row

    replay = dict(metadata or {})

    replay.update({
        "episode_id": db_id,
        "episode_name": name,
        "env_id": env_id,
        "seed": seed,
        "gymnasium_version": gym_version,
        "outcome": outcome,
        "steps": replay.get("steps", total_steps),
        "total_steps": total_steps,
        "total_reward": total_reward,
        "fps": fps or 50,
        "terrain": terrain,
        "constants": constants,
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
    })

    # Preserve original policy metadata if available.
    if "policy" not in replay:
        replay["policy"] = {
            "label": policy_text or "unknown",
            "type": "unknown"
        }

    required = [
        "constants",
        "terrain",
        "obs",
        "actions",
        "rewards"
    ]

    missing = [
        key for key in required
        if replay.get(key) is None
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Stored replay lacks viewer fields: "
                + ", ".join(missing)
            )
        )

    return replay
