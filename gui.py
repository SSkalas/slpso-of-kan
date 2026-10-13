import numpy as np
import itertools
import time
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt

from kan import *
from pso import *
from file_io import *
from screeninfo import get_monitors

NEURON_LABELS = {
    "Pendulum-v1": {
        "inputs":  ["cos(x)", "sin(x)", "angular velocity"],
        "outputs": ["torque"],
    },
    "LunarLander-v3": {
        "inputs":  ["x", "y", "x velocity", "y velocity", "angle",
                    "angular velocity", "left leg contact", "right leg contact"],
        "outputs": ["do nothing", "fire left", "fire main", "fire right"],
    },
    "CartPole-v1": {
        "inputs":  ["cart position", "cart velocity", "pole angle", "pole angular velocity"],
        "outputs": ["push left", "push right"],
    },
    "Acrobot-v1": {
            "inputs":  ["cos(x1)", "sin(x1)", "cos(x2)", "sin(x2)", "angular vel x1", "angular vel x2"],
            "outputs": ["apply -1 torque", "apply 0 torque", "apply 1 torque"],
    },
    "MountainCarContinuous-v0": {
            "inputs":  ["position", "velocity"],
            "outputs": ["force"],
    },
}

VIDEO_HEIGHT_FRAC = 0.6
LEFT_FRAC = 0.6
VIDEO_MARGIN = 40

class App:
    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.window_tag = "primary_window"
        self.text_tag = "main_text"
        self.child_window_tag1 = "child_window1"
        self.child_window_tag2 = "child_window2"
        self.video_area_tag = "video_area"
        self.sim_speed = 1.0
        self.last_step_time = 0.0
        self.split = LEFT_FRAC     

        self.viewport = None
        self.is_fullscreen = False
        self.windowed_state = None

        self.model = None
        self.kan_plot = None
        self.env_video = None
        self.obs_scale = None
        self.discrete = None

        self.loading_run = False

        self.run_loader = FileDialog(width=800, height=400, on_select=self.load_run)
        self.speed_slider = Slider(
            label="Simulation Speed",
            min_value=0.1,
            max_value=5.0,
            default_value=1.0,
            on_change=self.set_sim_speed
        )
        self.split_slider = Slider(
            label="Split Size",
            min_value=0.2,
            max_value=0.8,
            default_value=LEFT_FRAC,
            on_change=self.set_split,
        )
        self.reward_display = RewardDisplay()

    def current_monitor(self):
        x_pos, y_pos = dpg.get_viewport_pos()
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()

        cx = x_pos + width / 2
        cy = y_pos + height / 2

        monitors = get_monitors()
        for monitor in monitors:
            if monitor.x <= cx < monitor.x + monitor.width and monitor.y <= cy < monitor.y + monitor.height:
                return monitor
        return monitors[0]

    # Not really fullscreen, just borderless and max height+width
    def toggle_fullscreen(self):
        
        if not self.is_fullscreen:
            x_pos, y_pos = dpg.get_viewport_pos()
            self.windowed_state = (x_pos, y_pos, dpg.get_viewport_width(), dpg.get_viewport_height())
            monitor = self.current_monitor()
            dpg.configure_viewport(
                self.viewport,
                decorated=False,
                x_pos=monitor.x,
                y_pos=monitor.y,
                width=monitor.width,
                height=monitor.height
            )
        else:
            x_pos, y_pos, width, height = self.windowed_state
            dpg.configure_viewport(
                self.viewport,
                decorated=True,
                x_pos=x_pos,
                y_pos=y_pos,
                width=width,
                height=height
            )
        self.is_fullscreen = not self.is_fullscreen
        
        #dpg.toggle_viewport_fullscreen()

    def set_split(self, value) -> None:
        self.split = value
        self.apply_layout()
    
    def set_sim_speed(self, value):
        self.sim_speed = value
    
    def resize(self, sender=None, app_data=None) -> None:
        width = dpg.get_viewport_client_width()
        left_width = int(width * LEFT_FRAC)
        dpg.configure_item(self.child_window_tag1, width=left_width)
        self.resize_video()

    def on_viewport_resize(self, sender=None, app_data=None) -> None:
        self.apply_layout()

    def apply_layout(self) -> None:
        client_w = dpg.get_viewport_client_width()
        dpg.configure_item(self.child_window_tag1, width=int(client_w * self.split))
        self.resize_video()

    def resize_video(self) -> None:
        if self.env_video is None:
            return
        client_w = dpg.get_viewport_client_width()
        client_h = dpg.get_viewport_client_height()

        right_w = client_w - int(client_w * self.split)
        video_w = max(right_w - VIDEO_MARGIN, 50)
        video_h = max(int(client_h * VIDEO_HEIGHT_FRAC), 50)

        dpg.configure_item(self.env_video.tag("image"), width=video_w, height=video_h)
        
    def setup(self) -> None:
        dpg.create_context()
        self.viewport = dpg.create_viewport(title=self.title, width=self.width, height=self.height, resizable=True)
        dpg.setup_dearpygui()

        with dpg.window(label="main", tag=self.window_tag):
            self.run_loader.build()
            with dpg.viewport_menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Load Run...", callback=self.run_loader.show)
                with dpg.menu(label="View"):
                    dpg.add_menu_item(label="Fullscreen", shortcut="F11", callback=self.toggle_fullscreen)
                with dpg.menu(label="Simulation"):
                    self.speed_slider.build(parent=dpg.last_container())
                with dpg.menu(label="Split Size"):
                    self.split_slider.build(parent=dpg.last_container())
            with dpg.group(horizontal=True):
                dpg.add_child_window(width=self.width * LEFT_FRAC, height=-1, tag=self.child_window_tag1)
                with dpg.child_window(width=-1, height=-1, tag=self.child_window_tag2):
                    dpg.add_group(tag=self.video_area_tag)
                    self.reward_display.build(parent=self.child_window_tag2)
                    dpg.add_group(tag='stats_area')

        dpg.set_primary_window(self.window_tag, True)
        dpg.set_viewport_resize_callback(self.resize)
        self.load_run("runs/CartPole-v1_20260927-111136")
        self.apply_layout() 

    def load_run(self, dir):
        self.loading_run = True

        try:
            data, arrays = load_run(dir)
            architecture = data["architecture"]

            self.model = PSO_KAN(
                num_particles=1,
                num_params=architecture["num_params"],
                layer_sizes=architecture["layer_sizes"],
            )
            self.model.set_params(arrays["best_position"][None, :])

            dpg.delete_item(self.child_window_tag1, children_only=True)
            dpg.delete_item(self.video_area_tag, children_only=True)

            self.kan_plot = KAN_Plot(
                label=data["env_name"], width=-1, height=-1,
                model=self.model, coeffs=arrays["best_position"],
            )
            self.kan_plot.build(parent=self.child_window_tag1)

            if self.env_video is not None:
                self.env_video.close()  

            self.discrete = data["discrete"]
            self.env_video = Env_Video(env_name=data["env_name"], width=650, height=500, obs_scale=arrays["obs_scale"])
            self.env_video.build(parent=self.video_area_tag)
            self.reward_display.update(0.0, 0.0)

            dpg.delete_item("stats_area", children_only=True)
            self.stats_plot = StatsPlot(label="Stats", width=-1, height=-1,
                                        xs=range(len(arrays["best_fitness_history"])),
                                        ys=arrays["best_fitness_history"])
            self.stats_plot.build(parent="stats_area")
            self.resize_video()
        finally:
            self.loading_run = False

    def render_loop(self):
        if self.loading_run:
            dpg.render_dearpygui_frame()
            return

        if self.model is None or self.env_video is None:
            dpg.render_dearpygui_frame()
            return

        base_fps = self.env_video.env.metadata["render_fps"]
        step_interval = (1.0 / base_fps) / self.sim_speed
        now = time.perf_counter()

        if now - self.last_step_time >= step_interval:
            obs = self.env_video.obs
            self.kan_plot.update(obs)

            if self.discrete:
                action = np.argmax(self.model.forward(self.env_video.obs[None, :])[0])
            else:
                action = self.model.forward(self.env_video.obs[None, :])[0]
            self.env_video.step(action)
            self.reward_display.update(self.env_video.episode_reward, self.env_video.last_episode_reward)
            self.last_step_time = now

        dpg.render_dearpygui_frame()

    def update_text(self, value):
        dpg.set_value(self.text_tag, f"{value}")

    def run(self) -> None:
        self.setup()
        dpg.show_viewport()

        #dpg.toggle_viewport_fullscreen()
        while dpg.is_dearpygui_running():
            self.render_loop()
    
        dpg.destroy_context()

class Component:
    id_counter = itertools.count()
    def __init__(self, tag_prefix: str) -> None:
        self.id = next(Component.id_counter)
        self.tag_prefix = f"{tag_prefix}_{self.id}"

    def tag(self, name: str) -> str:
        return f"{self.tag_prefix}_{name}"

    def build(self, parents: str) -> None:
        raise NotImplementedError

class Env_Video(Component):
    def __init__(self, env_name: str, width: int, height: int, obs_scale: list[float]):
        super().__init__(tag_prefix="env_video")
        self.env_name = env_name
        self.width = width
        self.height = height
        self.env = gym.make(id=self.env_name, render_mode="rgb_array")
        self.env.metadata["render_fps"] = 30
        self.obs_scale = obs_scale
        self.obs, self.info = self.env.reset()
        self.obs /= self.obs_scale

        self.frame_height = None
        self.frame_width = None
        self.texture_tag = None

        self.episode_reward = 0.0
        self.last_episode_reward = 0.0

    def build(self, parent: str) -> None:
        frame = self.render_frame()

        with dpg.texture_registry():
            self.texture_tag = dpg.add_dynamic_texture(
                width=self.frame_width,
                height=self.frame_height,
                default_value=frame,
                tag=self.tag("texture"),
            )
            print(f"texture_tag = {self.texture_tag!r}, exists = {dpg.does_item_exist(self.texture_tag)}")

        dpg.add_image(
            self.texture_tag,
            parent=parent,
            width=self.width,
            height=self.height,
            tag=self.tag("image")
        )

    def render_frame(self):
        rgb_array = self.env.render()
        # rgb_array has shape (height, width, 3)
        # dearpygui needs shape (height * width * 4, ), extra alpha channel

        self.frame_height = rgb_array.shape[0]
        self.frame_width = rgb_array.shape[1]
        alphas = np.full(shape=(self.frame_height, self.frame_width, 1), fill_value=255, dtype=np.uint8)

        result = np.concatenate([rgb_array, alphas], axis=-1)
        return (result.astype(np.float32) / 255.0).flatten()

    def step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.obs /= self.obs_scale
        self.episode_reward += reward

        if terminated or truncated:
            self.last_episode_reward = self.episode_reward
            self.episode_reward = 0.0
            self.obs, self.info = self.env.reset()
            self.obs /= self.obs_scale

        dpg.set_value(self.texture_tag, self.render_frame())

    def close(self):
        self.env.close()

    def change_fps(self, mult):
        self.env.metadata["render_fps"] *= mult

class StatsPlot(Component):
    def __init__(self, label: str, width: int, height: int, xs, ys):
        super().__init__(tag_prefix="plot")
        self.label = label
        self.width = width
        self.heigt = height
        self.xs = xs
        self.ys = ys

    def build(self, parent: str) -> None:
        with dpg.theme() as self.line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

        with dpg.plot(label=self.label, width=self.width, height=self.heigt, parent=parent, tag=self.tag("stats_plot")):
            dpg.add_plot_axis(dpg.mvXAxis, label="Iteration")
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Best fitness", tag='yaxis')
            line_id = dpg.add_line_series(x=list(self.xs), y=list(self.ys), parent=yaxis)
            dpg.bind_item_theme(line_id, self.line_theme)

class KAN_Plot(Component):
    def __init__(self, label: str, width: int, height: int, model: PSO_KAN, coeffs: list[float]):
        super().__init__(tag_prefix="plot")
        self.label = label
        self.width = width
        self.heigt = height
        self.model = model
        self.coeffs = coeffs
        self.edges: list[KAN_Edge] = []

    def build(self, parent: str) -> None:
        with dpg.theme() as self.square_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, [255, 255, 255, 20], category=dpg.mvThemeCat_Plots
            )

        self.edges_by_layer = []
        with dpg.plot(label=self.label, width=self.width, height=self.heigt, parent=parent, tag=self.tag("plot")):

            self.xaxis = dpg.add_plot_axis(dpg.mvXAxis)
            self.yaxis = dpg.add_plot_axis(dpg.mvYAxis)

            coords, func_coords = self.model.get_coords()
            x_data = [c[0] for c in coords.values()]
            y_data = [c[1] for c in coords.values()]
            dpg.add_scatter_series(x_data, y_data, label="KAN", parent=self.yaxis)

            labels = NEURON_LABELS.get(self.label, {})
            last = len(self.model.layer_sizes) - 1

            input_neurons  = [n for n, c in coords.items() if c[0] == 0]
            output_neurons = [n for n, c in coords.items() if c[0] == last]

            for neuron, text in zip(input_neurons, labels.get("inputs", [])):
                x, y = coords[neuron]
                dpg.add_plot_annotation(
                    label=text, default_value=(x, y),
                    offset=(-15, 0),    # pixels, pushes the text to the left
                    color=[255, 255, 255, 255], clamped=False,
                )

            for neuron, text in zip(output_neurons, labels.get("outputs", [])):
                x, y = coords[neuron]
                dpg.add_plot_annotation(
                    label=text, default_value=(x, y),
                    offset=(15, 0),                     # pushes the text to the right
                    color=[255, 255, 255, 255], clamped=False,
    )

            half_w = 0.025
            half_h = 1.25
            num_coeffs = self.model.num_params
            offset = 0

            for i in range(len(self.model.layer_sizes) - 1):
                layer_edges = []
                current_layer = [n for n, c in coords.items() if c[0] == i]
                next_layer = [n for n, c in coords.items() if c[0] == i + 1]
                func_layer = [f for f, c in func_coords.items() if c[0] == i + 0.5]
        
                func_counter = 0
                for neuron_a in current_layer:
                    for neuron_b in next_layer:
                        func_id = func_layer[func_counter]
                        func_counter += 1
        
                        params = self.coeffs[offset:offset + num_coeffs]
                        offset += num_coeffs
        
                        xa, ya = coords[neuron_a]
                        xf, yf = func_coords[func_id]
                        xb, yb = coords[neuron_b]
        
                        edge = KAN_Edge(xa, ya, xb, yb, xf, yf, params)
                        edge.build(parent=self.yaxis, square_theme=self.square_theme)
                        self.edges.append(edge)
                        layer_edges.append(edge)
                self.edges_by_layer.append(layer_edges)

    def update(self, obs) -> None:
        h = np.asarray(obs, dtype=np.float64)
        sizes = self.model.layer_sizes
        for edges, n_in, n_out in zip(self.edges_by_layer, sizes[:-1], sizes[1:]):
            x = np.tanh(h)                       
            out = np.zeros(n_out)
            for a in range(n_in):
                for b in range(n_out):
                    edge = edges[a * n_out + b]
                    y = edge.evaluate(x[a])
                    #edge.set_activation(y)
                    out[b] += y
            h = out                  

class KAN_Edge(Component):
    def __init__(self, xa, ya, xb, yb , xf, yf, params, half_w=0.025, half_h=1.25):
        super().__init__(tag_prefix="edge")
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xf, self.yf = xf, yf
        self.params = params
        self.half_w = half_w
        self.half_h = half_h

        self.line_ids: list[int] = []
        self.square_id: int | None = None
        self.spline_id: int | None = None

    def build(self, parent: str, square_theme) -> None:
        with dpg.theme() as self.edge_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                self.edge_color = dpg.add_theme_color(
                    dpg.mvPlotCol_Line, [255, 255, 255, 255],
                    category=dpg.mvThemeCat_Plots,
                )   
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
                )
        with dpg.theme() as self.axis_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 0, 0, 255],
                                    category=dpg.mvThemeCat_Plots)
                
        self.line_ids.append(dpg.add_line_series(
            [self.xa, self.xf - self.half_w], [self.ya, self.yf], parent=parent
        ))
        self.line_ids.append(dpg.add_line_series(
            [self.xf + self.half_w, self.xb], [self.yf, self.yb], parent=parent
        ))
        for id in self.line_ids:
            dpg.bind_item_theme(id, self.edge_theme)

        x_square = [
            self.xf - self.half_w, self.xf + self.half_w, self.xf + self.half_w, 
            self.xf - self.half_w, self.xf - self.half_w
        ]
        y_sqaure = [
            self.yf - self.half_h, self.yf - self.half_h, self.yf + self.half_h, 
            self.yf + self.half_h, self.yf - self.half_h
        ]
        
        self.square_id = dpg.add_line_series(x_square, y_sqaure, parent=parent)
        dpg.bind_item_theme(self.square_id, square_theme)

        cheby_x, cheby_y, axis_x, axis_y = self.compute_spline()
        self.spline_id = dpg.add_line_series(cheby_x, cheby_y, parent=parent)
        if axis_x is not None:
            self.axis_id = dpg.add_line_series(axis_x, axis_y, parent=parent)
            dpg.bind_item_theme(self.axis_id, self.axis_theme)

        self.cheby = np.polynomial.chebyshev.Chebyshev(self.params)
        ys = self.cheby(np.linspace(-1, 1, 300))
        self.range = max(np.abs(ys).max(), 1e-8)

    def evaluate(self, x):
        return float(self.cheby(x))

    def set_activation(self, y: float) -> None:
        s = float(np.clip(y / self.range, -1, 1))      
        base = np.array([255, 255, 255])
        target = np.array([255, 0, 0]) 
        rgb = base + (target - base) * abs(s)
        dpg.set_value(self.edge_color, [*rgb.astype(int), 255])
    
    def compute_spline(self):
        cheby = np.polynomial.chebyshev.Chebyshev(self.params)
        x_data = np.linspace(-1, 1, 300)
        y_data = cheby(x_data)

        target_min_x, target_max_x = self.xf - self.half_w, self.xf + self.half_w
        target_min_y, target_max_y = self.yf - self.half_h, self.yf + self.half_h

        y_min, y_max = y_data.min(), y_data.max()
        span = max(y_max - y_min, 1e-8)          

        def to_plot_y(v):
            return target_min_y + (v - y_min) / span * (target_max_y - target_min_y)

        x_rescaled = np.interp(x_data, [-1, 1], [target_min_x, target_max_x])
        y_rescaled = to_plot_y(y_data)

        #x axis
        zero_y = float(np.clip(to_plot_y(0.0), target_min_y, target_max_y))
        axis_x = [target_min_x, target_max_x]
        axis_y = [zero_y, zero_y]
        return x_rescaled, y_rescaled, axis_x, axis_y


class Slider(Component):
    def __init__(self, label: str, min_value: float = 0.0, max_value: float = 10.0, default_value: float = 1.0,
                 on_change = None):
        super().__init__(tag_prefix="slider")
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = default_value
        self.on_change = on_change

    def build(self, parent: str) -> None:
        with dpg.group(parent=parent):
            dpg.add_text(self.label)
            dpg.add_slider_float(
                tag=self.tag("slider"),
                default_value=self.value,
                min_value=self.min_value,
                max_value=self.max_value,
                callback=self._on_change,
            )

    def _on_change(self, sender, app_data):
        self.value = app_data
        if self.on_change is not None:
            self.on_change(self.value)

class RewardDisplay(Component):
    def __init__(self):
        super().__init__(tag_prefix="reward_display")

    def build(self, parent: str) -> None:
        with dpg.group(parent=parent):
            dpg.add_text("Current reward: 0.00 | Last run total: 0.00", tag=self.tag("reward"))

    def update(self, current_reward: float, last_total_reward: float) -> None:
        dpg.set_value(self.tag("reward"), f"Current reward: {current_reward:.2f} | Last run total: {last_total_reward:.2f}")

class FileDialog(Component):
    def __init__(self, width, height, on_select = None):
        super().__init__(tag_prefix="run_loader")
        self.width = width
        self.height = height
        self.on_select = on_select

    def build(self, parent: str = None) -> None:
        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=self._on_select,
            tag=self.tag("dialog"),
            width=700,
            height=400
        )

    def show(self):
        dpg.show_item(self.tag("dialog"))

    def _on_select(self, sender, app_data):
        dir = app_data["file_path_name"]
        if self.on_select is not None:
            self.on_select(dir)


if __name__ == "__main__":
    App(title="PSO of KAN", width=1000, height=400).run()
