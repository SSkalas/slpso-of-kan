import time
import json
import numpy as np
from pathlib import Path

def save_run(root: str, agent: "RL_Agent"):
    run_dir = Path(root) / f"{agent.env_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)

    data = {
        "env_name": agent.env_name,
        "discrete": agent.discrete,
        "architecture": {
            "num_particles": agent.kan.num_particles,
            "layer_sizes": list(agent.kan.layer_sizes),
            "num_params": agent.kan.num_params,
            "total_params": agent.kan.total_params,
            "basis": agent.kan.basis
        }
    }
    (run_dir / "data.json").write_text(json.dumps(data, indent=2))

    np.savez_compressed(
        run_dir / "arrays.npz",
        best_position=agent.pso.best_position,
        best_fitness_history=np.asarray(agent.pso.history["best_fitness"]),
        mean_fitness_history=np.asarray(agent.pso.history["mean_fitness"]),
        best_position_history=np.asarray(agent.pso.history["best_position"]),
        positions=np.asarray(agent.pso.positions),
        obs_scale=np.asarray(agent.obs_scale)
    )
    return run_dir

def load_run(run_dir):
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "data.json").read_text())
    
    with np.load(run_dir / "arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}

    return data, arrays
