import numpy as np
import math
from tqdm import tqdm

class PSO:
    def __init__(self, num_particles: int, x_min: np.ndarray, x_max: np.ndarray, seed: int = None):
        if num_particles <= 0:
            raise ValueError("Number of particles cannot be negative or zero.")
        if len(x_min) <= 0 or len(x_max) <= 0:
            raise ValueError("Number of dimensions cannot be negative or zero.")
        if len(x_min) != len(x_max):
            raise ValueError(
                f"Dimension mismatch: x_min (len={len(x_min)}) and "
                f"x_max (len={len(x_max)}) must have the same length."
            )
        self.num_particles = num_particles
        self.num_dim = len(x_min)
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def rastrigin(x):
        n = len(x[0])
        return (10 * n) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

    def __repr__(self) -> str:
        return (
            f"ParticleSwarmOptimizer(num_particles={self.num_particles}, "
            f"x_min={self.x_min!r}, x_max={self.x_max!r}, seed={self.seed!r})"
        )
        
class ParticleSwarmOptimizer(PSO):
    def __init__(self, num_particles: int, x_min: np.ndarray, x_max: np.ndarray, seed: int = None):
        super().__init__(num_particles, x_min, x_max, seed)

        self.positions = self.rng.uniform(low=x_min, high=x_max, size=(self.num_particles, self.num_dim))
        self.p_best_positions = self.positions.copy()
        self.velocities = np.zeros(shape=(self.num_particles, self.num_dim))
        self.fitness = np.zeros(self.num_particles)
        self.p_best_fitness = np.zeros(self.num_particles)
        self.best_fitness = float('inf')
        self.best_position = self.positions[0].copy()

        self.history = {
            "best_fitness": [],
            "mean_fitness": [],
            "best_position": []
        }

    def get_local_best(self, num_neighbours):

        best_fitness = self.p_best_fitness.copy()
        best_positions = self.p_best_positions.copy()

        for shift in range(1, num_neighbours + 1):
            for direction in (shift, -shift):
                neighbour_fitness = np.roll(self.p_best_fitness, direction)
                neighbour_positions = np.roll(self.p_best_positions, direction, axis=0)

                better_mask = neighbour_fitness < best_fitness
                best_fitness = np.where(better_mask, neighbour_fitness, best_fitness)
                best_positions = np.where(better_mask[:, None], neighbour_positions, best_positions)

        return best_fitness, best_positions


    def optimize(self, num_iterations, fitness_function, cognitive_component=1.49445,
                social_component=1.49445, max_inertia=0.729, min_inertia=0.729, early_stopping=2000
        ):

        # First, calculate and store each particle's current fitness
        self.fitness = fitness_function(self.positions)

        self.p_best_fitness = self.fitness.copy()

        no_improvement = 0
        prev_g_best_fitness = np.min(self.p_best_fitness)
        pbar = tqdm(range(num_iterations), desc="PSO")
        for it in pbar:

            # Set the personal best position
            p_best_mask = self.fitness < self.p_best_fitness

            # Only the p_best positions whose fitness improved will get updated
            self.p_best_positions[p_best_mask] = self.positions[p_best_mask]
            self.p_best_fitness[p_best_mask] = self.fitness[p_best_mask]


            # Update global best position and fitness
            best_index = np.argmin(self.p_best_fitness)
            if self.p_best_fitness[best_index] < self.best_fitness:
                self.best_position = self.p_best_positions[best_index].copy()
                self.best_fitness = self.p_best_fitness[best_index]

            # Update local best position and fitness
            _, l_best_position = self.get_local_best(num_neighbours=1)


            # Inertia (used in velocity update) decreases linearly
            inertia = ((max_inertia - min_inertia) * ((num_iterations - it) / num_iterations)) + min_inertia


            # Update particle positions and velocities
            r1 = self.rng.uniform(low=0, high=1, size=(self.num_particles, self.num_dim))
            r2 = self.rng.uniform(low=0, high=1, size=(self.num_particles, self.num_dim))

            self.velocities = (
                (self.velocities * inertia) +
                cognitive_component * r1 * (self.p_best_positions - self.positions) +
                social_component * r2 * (l_best_position- self.positions)
            )

            # Update each particle's position using the new velocities and clip so that it stays in bounds
            # Max velocity limit (typically 10-20% of spatial range)
            v_max = 0.2 * (self.x_max[0] - self.x_min[0])
            self.velocities = np.clip(self.velocities, -v_max, v_max)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.x_min, self.x_max)

            # Finally, update each particle's new fitness
            self.fitness = fitness_function(self.positions)

            self.history["best_fitness"].append(self.best_fitness)
            self.history["mean_fitness"].append(np.mean(self.fitness))
            self.history["best_position"].append(self.best_position.copy())

            pbar.set_postfix({
                "best_fitness": f"{self.best_fitness:.1f}"
            })

def standard_params(n, M=100, alpha=0.5, beta=0.01):
    m = M + n // 10
    eps = beta * n / M
    gamma = alpha * np.log(np.ceil(n / M))
    return m, eps, gamma

class SLPSO(PSO):
    def __init__(self, num_particles: int, x_min: np.ndarray, x_max: np.ndarray, seed: int = None):
        super().__init__(num_particles, x_min, x_max, seed)

        # Randomly initialize particles
        self.positions = self.rng.uniform(
            low=self.x_min,
            high=self.x_max,
            size=(self.num_particles, self.num_dim)
        )

        self.fitness = np.zeros(self.num_particles)
        self.delta_positions = np.zeros((self.num_particles, self.num_dim))

        self.history = {
            "best_fitness": [],
            "mean_fitness": [],
            "best_position": []
        }

        self.best_position = None
        self.best_fitness = np.inf

    def optimize(
        self,
        num_iterations,
        fitness_function,
        gamma,
        epsilon
    ):

        learning_probs = np.array([
            (1 - (i - 1)/self.num_particles)
            ** (
                gamma
            )
            for i in range(1, self.num_particles + 1)
        ])
        learning_probs[-1] = 0.0


        pbar = tqdm(range(num_iterations), desc="SLPSO")
        for it in pbar:
            #Evaluate particles
            self.fitness = fitness_function(self.positions)

            # Rank particles from worst to best
            sort_indices = np.argsort(self.fitness)[::-1]
            self.positions = self.positions[sort_indices]
            self.fitness = self.fitness[sort_indices]
            self.delta_positions = self.delta_positions[sort_indices]

            # Update global best
            if self.fitness[-1] < self.best_fitness:
                self.best_fitness = self.fitness[-1]
                self.best_position = self.positions[-1].copy()
                #print(f"\nNew Best! {self.best_fitness}")


            mean_position = self.positions.mean(axis=0)
            teachers = np.array([self.rng.integers(low=i, high=self.num_particles, size=self.num_dim) for i in range(1, self.num_particles)])
            teachers = np.append(teachers, np.array([[self.num_particles - 1] * self.num_dim]), axis=0)

            col_indices = np.arange(self.num_dim)
            teachers_positions = self.positions[teachers, col_indices]

            # Decide which particles learn
            random_probs = self.rng.random(self.num_particles)
            learn_mask = random_probs <= learning_probs

            imitation_components = teachers_positions - self.positions
            social_components = mean_position - self.positions
            r1 = self.rng.uniform(low=0, high=1, size=(self.num_particles, self.num_dim))
            r2 = self.rng.uniform(low=0, high=1, size=(self.num_particles, self.num_dim))
            r3 = self.rng.uniform(low=0, high=1, size=(self.num_particles, self.num_dim))

            new_delta = (
                r1[learn_mask] * self.delta_positions[learn_mask] +
                r2[learn_mask] * imitation_components[learn_mask] +
                r3[learn_mask] * epsilon * social_components[learn_mask]
            )

            self.delta_positions[learn_mask] = new_delta
            self.positions[learn_mask] += self.delta_positions[learn_mask]
            self.positions = np.clip(self.positions, self.x_min, self.x_max)

            self.history["best_fitness"].append(self.best_fitness)
            self.history["mean_fitness"].append(np.mean(self.fitness))
            self.history["best_position"].append(self.best_position.copy())

            pbar.set_postfix({
                "best_fitness": f"{self.best_fitness:.1f}"
            })

        return self.best_position, self.best_fitness

    @staticmethod
    def rastrigin(x):
        n = x.shape[1]

        return (
            10 * n
            + np.sum(
                x**2 - 10 * np.cos(2 * np.pi * x),
                axis=1
            )
        )



