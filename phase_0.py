
import gymnasium as gym
import time


# Change this to 0.03 or 0.05 to slow down playback.
PLAYBACK_DELAY = 0.01
MAX_STEPS = 1000


def choose_action(test_name, env):
    """Select an action for the current diagnostic test."""

    if test_name == "Random Agent":
        return env.action_space.sample()

    if test_name == "No Thrust":
        return 0

    if test_name == "Main Engine Only":
        return 2

    raise ValueError(f"Unknown test: {test_name}")


def run_simulation(test_name, seed):
    print("\n" + "=" * 50)
    print(f"TEST: {test_name}")
    print("=" * 50)

    env = gym.make("LunarLander-v3", render_mode="human")

    try:
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        terminated = False
        truncated = False

        for step in range(MAX_STEPS):
            action = choose_action(test_name, env)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            time.sleep(PLAYBACK_DELAY)

            if terminated or truncated:
                break

        print(f"Test: {test_name}")
        print(f"Steps: {step + 1}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Final state: {observation}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    finally:
        env.close()


def main():
    tests = [
        ("Random Agent", 42),
        ("No Thrust", 43),
        ("Main Engine Only", 44),
    ]

    print("LUNAR LANDER: PHASE 0 EXPERIMENTS")
    print("Three diagnostic simulations will run sequentially.")
    print("Close/finish each run before starting the next.")

    for index, (test_name, seed) in enumerate(tests):
        input(
            f"\nPress ENTER to start test {index + 1}/"
            f"{len(tests)}: {test_name}..."
        )
        run_simulation(test_name, seed)

    print("\nAll diagnostic simulations completed.")


if __name__ == "__main__":
    main()
