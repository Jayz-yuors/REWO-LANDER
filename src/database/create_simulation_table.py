
from db_config import create_connection


def create_simulation_table():
    conn = create_connection()

    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rl_simulation_episodes (

                episode_id BIGSERIAL PRIMARY KEY,

                episode_name VARCHAR(150) UNIQUE NOT NULL,

                env_id VARCHAR(100) NOT NULL,
                seed INTEGER,
                policy VARCHAR(100),
                gymnasium_version VARCHAR(50),

                outcome VARCHAR(30),
                total_steps INTEGER NOT NULL,
                total_reward DOUBLE PRECISION,

                fps INTEGER,

                terrain JSONB,
                constants JSONB,

                observations JSONB NOT NULL,
                actions JSONB NOT NULL,
                rewards JSONB,

                metadata JSONB,

                created_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Useful filters for browsing simulation runs
        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_rl_episodes_outcome
            ON rl_simulation_episodes(outcome);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_rl_episodes_created
            ON rl_simulation_episodes(created_at DESC);
        """)

        conn.commit()

        print("Simulation table created successfully.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    create_simulation_table()
