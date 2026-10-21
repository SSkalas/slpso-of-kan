import numpy as np
import matplotlib.pyplot as plt
from file_io import load_run
from pso import SLPSO, ParticleSwarmOptimizer
from kan import PSO_KAN
from agent import RL_Agent

def main():
    BUDGET = 1_000
    NUM_EPISODES = 5
    PARTICLES = 50
    EVAL_SEEDS = range(1000000, 1000050)
    GAMMAS = 0.2
    EPSILONS = 0.01

    kan = PSO_KAN(
        num_particles=PARTICLES,
        num_params=4,
        layer_sizes=[8, 4],
        basis="chebyshev"
    )

    pso = SLPSO(
        num_particles=PARTICLES,
        x_min=[-1] * kan.total_params,
        x_max=[1] * kan.total_params
    )

    agent = RL_Agent(
        env_name="LunarLander-v3",
        num_envs=PARTICLES,
        kan=kan,
        pso=pso,
        obs_scale=[1]*8,
        discrete=True,
        num_episodes=NUM_EPISODES
    )
    run_dir = agent.train(num_iterations=BUDGET // NUM_EPISODES // PARTICLES, gamma=GAMMAS, epsilon=EPSILONS)
    agent.display(run_dir)



if __name__ == "__main__":
    main()