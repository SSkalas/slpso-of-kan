import numpy as np
import matplotlib.pyplot as plt
from file_io import load_run
from pso import SLPSO, ParticleSwarmOptimizer
from kan import PSO_KAN
from agent import RL_Agent

def main():
    kan = PSO_KAN(
        num_particles=50,
        num_params=4,
        layer_sizes=[8, 4],
        basis="chebyshev"
    )

    pso = ParticleSwarmOptimizer(
        num_particles=50,
        x_min=[-1] * kan.total_params,
        x_max=[1] * kan.total_params,
        seed=42
    )

    agent = RL_Agent(
        env_name="LunarLander-v3",
        num_envs=50,
        kan=kan,
        pso=pso,
        obs_scale=[1]*8,
        discrete=True,
        num_episodes=10,
    )
    #agent.train(num_iterations=80, gamma=0.05, epsilon=0.01)
    
    _, arrays_slpso = load_run("runs/LunarLander-v3_20261012-113203")
    _, arrays_pso = load_run("runs/LunarLander-v3_20261012-113727")
    history_slpso = arrays_slpso["best_fitness_history"]
    history_pso = arrays_pso["best_fitness_history"]
    plt.plot(history_slpso, label='SL-PSO')
    plt.plot(history_pso, label='PSO')
    plt.legend()
    plt.show()

    pso_eval = agent.evaluate("runs/LunarLander-v3_20261012-113727", 20)
    print(np.mean(pso_eval))
    print(np.std(pso_eval))
    #agent.display("runs/LunarLander-v3_20261012-110018")
    
    


if __name__ == "__main__":
    main()