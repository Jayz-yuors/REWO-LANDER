# Rocket Lander 3D — Gymnasium + GLB visualizer

This starter project connects **Gymnasium `LunarLander-v3`** (the physics and episode backbone) to a **Three.js** 3D scene that replays recorded observations and actions. It uses the Apollo Lunar Module GLB you supplied.

## Important scientific limitation

The visualizer is a 3D *presentation of the Gymnasium trajectory*. The standard `LunarLander-v3` environment is a simplified **2D Box2D** simulation. Rendering its state with a 3D model does not make the physics 3D. This is a useful first milestone because it keeps the learning environment stable and separates simulation from presentation. A true 3D rigid-body lander would be a later environment project.

## Requirements

- Python 3.10+ (your existing Python virtual environment is fine)
- Your existing packages: `gymnasium[box2d]` and optionally `stable-baselines3`
- Node.js/npm
- Internet access once to install the frontend dependencies

## 1. Export a trajectory from Gymnasium

From this project folder, run:

```powershell
python scripts/export_trajectory.py --episodes 1 --policy random
```

To replay a trained Stable-Baselines3 model:

```powershell
python scripts/export_trajectory.py --episodes 1 --policy model --model path\to\your_model.zip
```

The script writes `visualizer/public/data/trajectory.json`. Random policy is for a visual smoke test only, not a meaningful landing controller.

## 2. Start the 3D viewer

```powershell
cd visualizer
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`). Keep the terminal running while you use the viewer.

## Viewer controls

- **Play/Pause** the recorded trajectory
- **Restart** playback
- **Playback speed** adjustment
- Live observation values, action, reward, and leg-contact indicators
- Orbit the scene with the mouse; scroll to zoom

## Coordinate mapping

The exporter records all eight Gymnasium observation values and the action/reward per step. The viewer maps the 2D environment's x/y/angle into the 3D scene. The vertical and horizontal scales are visual scaling factors, not physical units. Engine plumes are visual indicators based on the discrete action.

## Suggested next steps

1. Run the random-policy replay to confirm the GLB orientation and scale.
2. Train PPO using your existing setup, then export a trained episode.
3. Add a trajectory selector for successful / failed evaluation episodes.
4. If the research objective requires physically realistic 3D dynamics, implement a separate 3D Gymnasium environment and validate its equations, contacts, thrust, and reward function. Do not describe the current visual replay as 3D physics.
