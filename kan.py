import numpy as np
import gymnasium as gym

class PSO_KAN:
    def __init__(self, num_particles: int, num_params: int, layer_sizes: list[int], basis="chebyshev"):
        self.num_particles = num_particles
        self.num_params = num_params
        self.layer_sizes = layer_sizes
        self.basis=basis

        self.total_params = 0
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            num_inputs = self.layer_sizes[i]
            num_outputs = self.layer_sizes[i + 1]
            self.total_params += self.num_params * num_inputs * num_outputs
            self.layers.append(
                PSO_KAN_Layer(
                    self.num_particles, self.num_params, dims=(num_inputs, num_outputs), basis=basis
                )
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def set_params(self, positions: np.ndarray):
        start_idx = 0
        for layer in self.layers:
            # Calculate how many parameters this layer requires
            layer_params_size = layer.num_inputs * layer.num_outputs * layer.num_params
            end_idx = start_idx + layer_params_size
            
            # Extract and reshape the parameters for this layer
            layer_positions = positions[:, start_idx:end_idx]
            layer.params = layer_positions.reshape(
                self.num_particles, layer.num_inputs, layer.num_outputs, self.num_params
            )
            
            start_idx = end_idx

    def get_coords(self):
        neuron_coords = {}
        function_coords = {}
        y_max = 100

        neuron_counter = 0
        for i in range(len(self.layer_sizes)):
            num_neurons = self.layer_sizes[i]
            for j in range(self.layer_sizes[i]):
                spacing = y_max / (num_neurons + 1)
                neuron_coords[neuron_counter] = [i, (j + 1) * spacing]
                neuron_counter += 1

        function_counter = 0
        for i in range(len(self.layer_sizes) - 1):
            num_functions = self.layer_sizes[i] * self.layer_sizes[i+1]
            spacing = y_max / (num_functions + 1)
            x_coord = (i + (i + 1)) / 2
            for j in range(num_functions):
                function_coords[function_counter] = [x_coord, (j+1) * spacing]
                function_counter += 1
                
        return neuron_coords, function_coords

    def __repr__(self):
        return (f"PSO_KAN(num_particles='{self.num_particles}', num_params='{self.num_params}', " +
                f"layer_sizes='{self.layer_sizes}', total_params='{self.total_params}', " +
                f"layers='{self.layers}')")
            
class PSO_KAN_Layer:
    def __init__(self, num_particles: int, num_params: int, dims: tuple[int, int], degree: int = 3, basis="chebyshev"):
            self.num_particles = num_particles
            # num_params must equal degree + 1 
            self.num_params = num_params 
            self.dims = dims
            self.degree = degree
            self.basis=basis

            self.num_inputs = dims[0]
            self.num_outputs = dims[1]
            self.params = np.zeros(
                shape=(self.num_particles, self.num_inputs, self.num_outputs, self.num_params)
            )

    def _evaluate_chebyshev(self, x: np.ndarray) -> np.ndarray:
        # C shape: (num_particles, num_inputs, degree + 1)
        C = np.zeros(x.shape + (self.degree + 1,))
        
        # T_0(x) = 1
        C[..., 0] = 1.0
        
        if self.degree >= 1:
            # T_1(x) = x
            C[..., 1] = x
            
        # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        for i in range(2, self.degree + 1):
            C[..., i] = 2 * x * C[..., i - 1] - C[..., i - 2]
            
        return C

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.tanh(x)
        if self.basis == "chebyshev":
            C = self._evaluate_chebyshev(x)
            return np.einsum('pik,pijk->pj', C, self.params)
        elif self.basis == "bspline":
            return self.forward_bspline(x)
        else:
            raise ValueError(f"unknown basis: {self.basis}")

    def _bspline_knots(self, num_ctrl_points, degree):
        num_knots = num_ctrl_points + degree + 1
        num_internal = num_knots - 2 * (degree + 1)
        if num_internal > 0:
            internal = np.linspace(-1, 1, num_internal + 2)[1:-1]
        else:
            internal = np.array([])
        return np.concatenate([
            np.full(degree + 1, -1.0),
            internal,
            np.full(degree + 1, 1.0),
        ])

    def _evaluate_bspline_basis(self, x, knots, degree, num_basis):
        num_knots = len(knots)

        # Degree 0: indicator functions over each knot span.
        N = []
        for i in range(num_knots - 1):
            left, right = knots[i], knots[i + 1]
            if i == num_knots - 2:  # last span includes the right edge (x == 1.0)
                Ni = ((x >= left) & (x <= right)).astype(np.float64)
            else:
                Ni = ((x >= left) & (x < right)).astype(np.float64)
            N.append(Ni)

        # Recurrence up to the target degree.
        for d in range(1, degree + 1):
            new_N = []
            for i in range(len(N) - 1):
                denom1 = knots[i + d] - knots[i]
                term1 = (x - knots[i]) / denom1 * N[i] if denom1 != 0 else np.zeros_like(x)

                denom2 = knots[i + d + 1] - knots[i + 1]
                term2 = (knots[i + d + 1] - x) / denom2 * N[i + 1] if denom2 != 0 else np.zeros_like(x)

                new_N.append(term1 + term2)
            N = new_N

        return np.stack(N[:num_basis], axis=-1)

    def forward_bspline(self, x: np.ndarray) -> np.ndarray:
        knots = self._bspline_knots(self.num_params, self.degree)
        B = self._evaluate_bspline_basis(x, knots, self.degree, self.num_params)  # (p, i, num_params)
        y = np.einsum('pik,pijk->pj', B, self.params)
        return y

    
    def __repr__(self):
        return (f"PSO_KAN_Layer(num_particles='{self.num_particles}', num_params='{self.num_params}', " +
                f"dims='{self.dims}')")