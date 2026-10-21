import numpy as np
import matplotlib.pyplot as plt
from file_io import load_run
from pso import SLPSO, ParticleSwarmOptimizer
from kan import PSO_KAN
from agent import RL_Agent

def main():
    results = []
    BUDGET = 50_000
    NUM_EPISODES = 5
    PARTICLES = 50
    EVAL_SEEDS = range(1000000, 1000050)
    GAMMAS = 0.2
    EPSILONS = 0.01
    SEEDS = range(8)
    for seed in SEEDS:
        kan = PSO_KAN(
            num_particles=PARTICLES,
            num_params=4,
            layer_sizes=[8, 4],
            basis="chebyshev"
        )

        pso = SLPSO(
            num_particles=PARTICLES,
            x_min=[-1] * kan.total_params,
            x_max=[1] * kan.total_params,
            seed=seed
        )

        agent = RL_Agent(
            env_name="LunarLander-v3",
            num_envs=PARTICLES,
            kan=kan,
            pso=pso,
            obs_scale=[1]*8,
            discrete=True,
            num_episodes=NUM_EPISODES,
            seed=seed
        )
        run_dir = agent.train(num_iterations=BUDGET // NUM_EPISODES // PARTICLES, gamma=GAMMAS, epsilon=EPSILONS)
        rewards = agent.evaluate(run_dir, EVAL_SEEDS)
        print(f"SLPSO {seed} {np.mean(rewards):.2f} {np.std(rewards):.2f}")
        results.append((seed, np.mean(rewards), np.std(rewards)))
        np.savetxt("slpso_results.csv", results, delimiter=",", header="seed,mean,std")

    '''
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
    '''
    


if __name__ == "__main__":
    main()