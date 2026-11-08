# Apollo 3D replay: corrected run instructions

The Pygame window is Gymnasium's native 2D renderer. Do **not** run a script with
`render_mode="human"` if you want only the Apollo 3D view.

## 1. Generate automated thruster trajectories (no Pygame window)

From the `13.Rocket_RL` project root, with your existing Python virtual environment active:

```powershell
python test_thrusters.py
```

The script runs Gymnasium `LunarLander-v3`/Box2D with fixed, automated action sequences.
It saves one replay JSON per test under `results/raw/replays/`. Edit `TESTS_TO_RUN` near
the top of `test_thrusters.py` to choose tests before running. There are no keyboard
controls during simulation. All tests use the same seed to make starting conditions
more comparable.

## 2. Open the Apollo GLB viewer

From the same project root, run:

```powershell
python -m http.server 8000
```

Open this URL in your browser:

`http://localhost:8000/web/replay3d/index.html`

The viewer loads `web/replay3d/assets/Apollo_Lunar_Module.glb` and the sample replay by
default. To inspect a generated thruster test, click **Load JSON…** in the viewer and
select one of `results/raw/replays/thruster_*.json`.

## What this does (and does not do)

- Gymnasium/Box2D is the only source of simulation physics, observations, rewards, and
  episode termination. The Three.js viewer reads those recorded values and animates the
  supplied Apollo Lunar Module model, engine plumes, terrain, and trail.
- The GLB is a visual representation, not a body that Gymnasium directly simulates.
- The physics remain the original 2D LunarLander physics represented in a 3D scene; this
  is not a 3D rigid-body simulation.
- The viewer imports Three.js and the Draco decoder from jsDelivr, so the first load needs
  internet access. Serve the files over HTTP; do not double-click the HTML file.
