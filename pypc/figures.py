import os
import plotly.graph_objs as go
# from dash import Dash, html, dcc
import numpy as np
import pandas as pd
import neptune.new as neptune

from collections import namedtuple
from operator import itemgetter

def rgb_to_rgba(rgb_value, alpha):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"

# Colours for plots
colours = {
    'white': 'rgb(255, 255, 255)',
    'grey75': 'rgb(191, 191, 191)',
    'grey50': 'rgb(127, 127, 127)',
    'grey25': 'rgb(63, 63, 63)',
    'black': 'rgb(0, 0, 0)',
    'darkred': 'rgb(127, 0, 0)',
    'red': 'rgb(255, 0, 0)',
    'orange': 'rgb(255, 127, 0)',
    'yellow': 'rgb(255, 255, 0)',
    'green': 'rgb(0, 255, 0)',
    'darkgreen': 'rgb(0, 127, 0)',
    'cyan': 'rgb(0, 255, 255)',
    'blue': 'rgb(0, 0, 255)',
    'darkblue': 'rgb(0, 0, 127)',
    'magenta': 'rgb(255, 0, 255)',
}

# Groups of runs to be added to dataset
RunGroup = namedtuple("RunGroup", ["start_run", "num_runs", "title", "colour", "x_value"])


class NickFig:
    def __init__(self, fig_type, groups, fig_num, title, yaxis_title, yaxis_scale, data_field, xaxis_title, project_name, num_datapoints, xaxis_range, yaxis_range, path, filename, showlegend, font, width, height, show_fig=False, save_fig=False):
        self.fig_type = fig_type
        self.groups = groups
        self.fig_num = fig_num
        self.title = title
        self.yaxis_title = yaxis_title
        self.data_field = data_field
        self.xaxis_title = xaxis_title
        self.project_name = project_name
        self.num_datapoints = num_datapoints
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.yaxis_scale = yaxis_scale
        self.path = path
        self.filename = filename
        self.showlegend = showlegend
        self.font = font
        self.width = width
        self.height = height
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.stats = pd.DataFrame()

        self.fig = go.Figure()

        self.format_figure()
        if self.fig_type == "1d_multiple_curves":
            self.calc_stats()
        elif self.fig_type == "2d":
            self.calc_stats_2dscan()
        else:
            assert False, "Select a valid figure type!"

        self.draw_traces()
        if self.show_fig:
            self.show_figure()
        if self.save_fig:
            self.save_figure()

    def get_runs(self, group):
        df = pd.DataFrame()
        for run in range(group.start_run, group.start_run + group.num_runs):
            run_id = f"PYPC-{run}"
            # print(f"{run_id}")
            run = neptune.init_run(
                project="lasermanick/PYPC",
                mode="read-only",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMDVhMjgyMi01ZjU0LTQ3NDMtODcwOS1jNjlmNGNjNDRhYTEifQ==",
                with_id=run_id,
            )
            df.insert(df.shape[1], run_id, run[self.data_field].fetch_values(include_timestamp=False)["value"])
        return df

    def calc_stats(self):
        self.stats.insert(self.stats.shape[1], self.xaxis_title,
                          np.arange(1, self.num_datapoints + 1))  # x-axis column
        for g in self.groups:
            df = self.get_runs(g)
            self.stats.insert(self.stats.shape[1], f"{g.title}m", df.mean(1) * self.yaxis_scale)
            self.stats.insert(self.stats.shape[1], f"{g.title}s", df.std(1) * self.yaxis_scale)
            self.stats.insert(self.stats.shape[1], f"{g.title}l", (df.mean(1) - df.std(1)) * self.yaxis_scale)
            self.stats.insert(self.stats.shape[1], f"{g.title}u", (df.mean(1) + df.std(1)) * self.yaxis_scale)

    def calc_stats_2dscan(self):
        for g in self.groups:  # For each group of runs in the figure
            df = self.get_runs(g)
            # Create the columns if needed
            if f"{g.title}m" not in self.stats.columns:
                self.stats.insert(self.stats.shape[1], f"{g.title}m", np.zeros(self.stats.shape[0]))
                self.stats.insert(self.stats.shape[1], f"{g.title}s", np.zeros(self.stats.shape[0]))
                self.stats.insert(self.stats.shape[1], f"{g.title}l", np.zeros(self.stats.shape[0]))
                self.stats.insert(self.stats.shape[1], f"{g.title}u", np.zeros(self.stats.shape[0]))
            # Create the row if needed
            if g.x_value not in self.stats.index:
                self.stats.loc[g.x_value] = np.zeros(self.stats.shape[1])
            # Add the stats to the stats dataframe
            self.stats.loc[g.x_value][f"{g.title}m"] = df.mean(1) * self.yaxis_scale
            self.stats.loc[g.x_value][f"{g.title}s"] = df.std(1) * self.yaxis_scale
            self.stats.loc[g.x_value][f"{g.title}l"] = (df.mean(1) - df.std(1)) * self.yaxis_scale
            self.stats.loc[g.x_value][f"{g.title}u"] = (df.mean(1) + df.std(1)) * self.yaxis_scale

        # Add x-axis column
        self.stats.insert(self.stats.shape[1], self.xaxis_title, self.stats.index)

    def draw_traces(self):
        # Extract traces (with unique titles) from the groups
        key = itemgetter(3)
        traces = {key(g): g for g in self.groups}.values()
        for t in traces:
            # Mean
            self.fig.add_trace(
                go.Scatter(
                    name=t.title,
                    x=self.stats[self.xaxis_title],
                    y=self.stats[f"{t.title}m"],
                    mode='lines',
                    line=dict(color=t.colour)
                )
            )
            # Upper bound
            self.fig.add_trace(
                go.Scatter(
                    name=f"{t.title} upper",
                    x=self.stats[self.xaxis_title],
                    y=self.stats[f"{t.title}u"],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            # Lower bound
            self.fig.add_trace(
                go.Scatter(
                    name=f"{t.title} lower",
                    x=self.stats[self.xaxis_title],
                    y=self.stats[f"{t.title}l"],
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=rgb_to_rgba(t.colour, 0.3),
                    fill='tonexty',
                    showlegend=False
                )
            )

    def format_figure(self):
        self.fig.update_layout(
            font=self.font,
            showlegend=self.showlegend,
            xaxis_title=self.xaxis_title,
            yaxis_title=self.yaxis_title,
            title=self.title,
            xaxis_range=self.xaxis_range,
            yaxis_range=self.yaxis_range,
            plot_bgcolor=colours['white'],
            width=self.width,
            height=self.height,
        )
        self.fig.update_xaxes(
            linecolor=colours['black'],
            showline=True,
            showgrid=False,
            linewidth=2,
        )
        self.fig.update_yaxes(
            linecolor=colours['black'],
            showline=True,
            showgrid=False,
            linewidth=2,
        )

    def show_figure(self):
        self.fig.show()

    def save_figure(self):
        figdir = self.path
        os.makedirs(figdir, exist_ok=True)
        self.fig.write_image(figdir + self.filename)

    def get_fig(self):
        return self.fig


testacc_fig_cfg = {
    "fig_type": "1d_multiple_curves",
    "title": "Title",
    "yaxis_title": "Test accuracy (%)",
    "data_field": "test/acc",
    "xaxis_title": "Epoch",
    "project_name": "PYPC",
    "num_datapoints": 20,
    "xaxis_range": [0, 20],
    "yaxis_range": [0, 100],
    "yaxis_scale": 100.0,
    "path": "figs/",
    "filename": "fig.svg",
    "showlegend": True,
    "font": dict(family="Arial", size=20, color=colours['black']),
    "width": 800,
    "height": 600,
}

fig_cfgs = {}

fig_num = "D(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} node structures 00"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(226, 6, "[10, 169, 729, 784]", colours['darkred'], None),
    RunGroup(232, 6, "[10, 144, 169, 784]", colours['red'], None),
    RunGroup(238, 6, "[10, 100, 300, 784]", colours['orange'], None),
    RunGroup(244, 6, "[10, 36, 169, 784]", colours['green'], None),
    RunGroup(250, 6, "[10, 10, 169, 729, 784]", colours['cyan'], None),
    RunGroup(256, 6, "[10, 676, 729, 784]", colours['blue'], None),
    RunGroup(262, 6, "[10, 36, 169, 729, 784]", colours['darkblue'], None),
]

fig_num = "D(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} node structures 01"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(268, 6, "0.001", colours['red'], None),
    RunGroup(262, 6, "0.003", colours['darkblue'], None),
    RunGroup(274, 6, "0.005", colours['green'], None),
]

fig_num = "D(c)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} node structures 02"
fc["filename"] = f"Fig {fig_num}.svg"
fc["num_datapoints"] = 40
fc["xaxis_range"] = [0, 40]
fc["groups"] = [
    RunGroup(280, 6, "[10, 169, 729, 784]", colours['red'], None),
    RunGroup(286, 6, "[10, 144, 169, 784]", colours['darkblue'], None),
    RunGroup(292, 6, "[10, 100, 300, 784]", colours['green'], None),
]

fig_num = "G(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} lr scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["num_datapoints"] = 40
fc["xaxis_range"] = [0, 40]
fc["groups"] = [
    # RunGroup(304, 6, "0.001", colours['red'], None),
    RunGroup(310, 6, "0.002", colours['darkred'], None),
    # RunGroup(316, 6, "0.003", colours['red'], None),
    # RunGroup(322, 6, "0.004", colours['red'], None),
    RunGroup(328, 6, "0.005", colours['red'], None),
    RunGroup(334, 6, "0.006", colours['orange'], None),
    RunGroup(340, 6, "0.008", colours['green'], None),
    RunGroup(346, 6, "0.010", colours['cyan'], None),
    RunGroup(364, 6, "0.015", colours['blue'], None),
    RunGroup(352, 6, "0.020", colours['darkblue'], None),
    # RunGroup(358, 6, "0.03", colours['red'], None),
]

fig_num = "G(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} dt scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(370, 6, "0.001", colours['red'], None),
    RunGroup(376, 6, "0.003", colours['orange'], None),
    RunGroup(382, 6, "0.01", colours['green'], None),
    RunGroup(388, 6, "0.03", colours['cyan'], None),
    RunGroup(394, 6, "0.1", colours['blue'], None),
]

fig_num = "I(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} training its scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(400, 6, "5", colours['red'], None),
    RunGroup(406, 6, "15", colours['orange'], None),
    RunGroup(412, 6, "50", colours['green'], None),
    RunGroup(418, 6, "150", colours['cyan'], None),
    RunGroup(424, 6, "500", colours['blue'], None),
]

fig_num = "I(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} test its scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(430, 6, "20", colours['red'], None),
    RunGroup(436, 6, "60", colours['orange'], None),
    RunGroup(442, 6, "200", colours['green'], None),
    RunGroup(448, 6, "600", colours['cyan'], None),
    RunGroup(454, 6, "2000", colours['blue'], None),
]

fig_num = "K(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} lr scan with extreme train and test its"
fc["filename"] = f"Fig {fig_num}.svg"
fc["num_datapoints"] = 40
fc["xaxis_range"] = [0, 40]
fc["groups"] = [
    RunGroup(460, 6, "0.001", colours['red'], None),
    RunGroup(466, 6, "0.002", colours['orange'], None),
    RunGroup(472, 6, "0.006", colours['green'], None),
    RunGroup(478, 6, "0.01", colours['cyan'], None),
    RunGroup(484, 6, "0.02", colours['blue'], None),
]

fig_num = "K(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} lr scan with extreme train and test its"
fc["filename"] = f"Fig {fig_num}.svg"
fc["num_datapoints"] = 40
fc["xaxis_range"] = [0, 40]
fc["groups"] = [
    RunGroup(490, 1, "0.001", colours['red'], None),
    RunGroup(492, 1, "0.002", colours['orange'], None),
    RunGroup(493, 1, "0.006", colours['green'], None),
    RunGroup(494, 1, "0.01", colours['cyan'], None),
    RunGroup(495, 1, "0.02", colours['blue'], None),
]

fig_num = "L"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} confirm prec change equiv to dt change"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(499, 6, "[10.0, 10.0, 10.0, 10.0]", colours['red'], None),
    RunGroup(505, 6, "[3.0, 3.0, 3.0, 3.0]", colours['orange'], None),
    RunGroup(511, 6, "[1.0, 1.0, 1.0, 1.0]", colours['green'], None),
    RunGroup(517, 6, "[0.3, 0.3, 0.3, 0.3]", colours['cyan'], None),
    RunGroup(523, 6, "[0.1, 0.1, 0.1, 0.1]", colours['blue'], None),
]

fig_num = "M(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(570, 6, "20", colours['darkred'], None),
    RunGroup(576, 6, "10", colours['red'], None),
    RunGroup(582, 6, "5", colours['orange'], None),
    RunGroup(588, 6, "2", colours['green'], None),
    RunGroup(594, 6, "1", colours['cyan'], None),
    RunGroup(600, 6, "0.5", colours['blue'], None),
    RunGroup(606, 6, "0.2", colours['darkblue'], None),
    RunGroup(612, 6, "0.1", colours['grey25'], None),
    RunGroup(618, 6, "0.05", colours['grey75'], None),
]

fig_num = "M(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} layer 2 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(624, 6, "20", colours['darkred'], None),
    RunGroup(630, 6, "10", colours['red'], None),
    RunGroup(636, 6, "5", colours['orange'], None),
    RunGroup(642, 6, "2", colours['green'], None),
    RunGroup(648, 6, "1", colours['cyan'], None),
    RunGroup(654, 6, "0.5", colours['blue'], None),
    RunGroup(660, 6, "0.2", colours['darkblue'], None),
    RunGroup(666, 6, "0.1", colours['grey25'], None),
    RunGroup(672, 6, "0.05", colours['grey75'], None),
]

fig_num = "M(c)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} layer 1 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(678, 6, "20", colours['darkred'], None),
    RunGroup(684, 6, "10", colours['red'], None),
    RunGroup(690, 6, "5", colours['orange'], None),
    RunGroup(696, 6, "2", colours['green'], None),
    RunGroup(702, 6, "1", colours['cyan'], None),
    RunGroup(708, 6, "0.5", colours['blue'], None),
    RunGroup(714, 6, "0.2", colours['darkblue'], None),
    RunGroup(720, 6, "0.1", colours['grey25'], None),
    RunGroup(726, 6, "0.05", colours['grey75'], None),
]

fig_num = "N(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} prec ratio by layer"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(1017, 1, "[1.0, 0.01, 0.01, 0.01]", colours['darkred'], None),
    RunGroup(1018, 1, "[1.0, 0.04, 0.02, 0.01]", colours['red'], None),
    RunGroup(1019, 1, "[1.0, 0.25, 0.05, 0.01]", colours['orange'], None),
    RunGroup(1020, 1, "[1.0, 1.0, 0.10, 0.01]", colours['green'], None),
    RunGroup(1021, 1, "[1.0, 4.0, 0.20, 0.01]", colours['cyan'], None),
    RunGroup(1022, 1, "[1.0, 25, 0.50, 0.01]", colours['blue'], None),
    RunGroup(1023, 1, "[1.0, 100, 1.0, 0.01]", colours['darkblue'], None),
]

fig_num = "N(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} prec ratio by layer"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(1024, 1, "[1.0, 0.01, 0.01, 0.01]", colours['darkred'], None),
    RunGroup(1025, 1, "[1.0, 0.01, 0.02, 0.04]", colours['red'], None),
    RunGroup(1026, 1, "[1.0, 0.01, 0.05, 0.25]", colours['orange'], None),
    RunGroup(1027, 1, "[1.0, 0.01, 0.10, 1.0]", colours['green'], None),
    RunGroup(1028, 1, "[1.0, 0.01, 0.20, 4.0]", colours['cyan'], None),
    RunGroup(1029, 1, "[1.0, 0.01, 0.50, 25]", colours['blue'], None),
    RunGroup(1030, 1, "[1.0, 0.01, 1.0, 100]", colours['darkblue'], None),
]

fig_num = "P(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} train vs test accuracy - test"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(732, 1, "0.001", colours['darkred'], None),
    RunGroup(734, 1, "0.003", colours['red'], None),
    RunGroup(735, 1, "0.006", colours['orange'], None),
    RunGroup(736, 1, "0.010", colours['green'], None),
    RunGroup(737, 1, "0.015", colours['cyan'], None),
    RunGroup(738, 1, "0.030", colours['blue'], None),
]

fig_num = "P(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["yaxis_title"] = "Training accuracy (%)"
fc["data_field"] = "train/acc"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} train vs test accuracy - train"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(732, 1, "0.001", colours['darkred'], None),
    RunGroup(734, 1, "0.003", colours['red'], None),
    RunGroup(735, 1, "0.006", colours['orange'], None),
    RunGroup(736, 1, "0.010", colours['green'], None),
    RunGroup(737, 1, "0.015", colours['cyan'], None),
    RunGroup(738, 1, "0.030", colours['blue'], None),
]

fig_num = "Q"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} effect of noise on accuracy"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(801, 1, "var = 4, cov = 1.0", colours['darkgreen'], None),
    RunGroup(841, 6, "var = 4, cov = 0.5", colours['green'], None),
    RunGroup(800, 1, "var = 1, cov = 1.0", colours['darkred'], None),
    RunGroup(813, 1, "var = 1, cov = 0.5", colours['red'], None),
    RunGroup(702, 6, "No noise", colours['grey25'], None),
]

fig_num = "R"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} bottom half noise, bottom half layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(817, 6, "0.1", colours['darkred'], None),
    RunGroup(823, 6, "0.2", colours['red'], None),
    RunGroup(829, 6, "0.5", colours['orange'], None),
    # RunGroup(835, 6, "0.8", colours['green'], None),
    RunGroup(841, 6, "1", colours['green'], None),
    RunGroup(847, 6, "2", colours['cyan'], None),
    RunGroup(853, 6, "5", colours['blue'], None),
    # RunGroup(859, 6, "8", colours['grey25'], None),
    RunGroup(865, 6, "10", colours['darkblue'], None),
    RunGroup(702, 6, "No noise", colours['grey25'], None),
]

fig_num = "S"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} bottom half noise, full layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(873, 1, "0.1", colours['darkred'], None),
    RunGroup(874, 1, "0.2", colours['red'], None),
    RunGroup(875, 1, "0.5", colours['orange'], None),
    # RunGroup(876, 1, "0.8", colours['green'], None),
    RunGroup(841, 6, "1", colours['green'], None),
    RunGroup(878, 1, "2", colours['cyan'], None),
    RunGroup(879, 1, "5", colours['blue'], None),
    # RunGroup(880, 1, "8", colours['grey25'], None),
    RunGroup(881, 1, "10", colours['darkblue'], None),
    RunGroup(702, 6, "No noise", colours['grey25'], None),
]

fig_num = "T"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} bottom half noise, full layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(883, 1, "0.1", colours['darkred'], None),
    RunGroup(884, 1, "0.2", colours['red'], None),
    RunGroup(885, 1, "0.5", colours['orange'], None),
    # RunGroup(886, 1, "0.8", colours['green'], None),
    RunGroup(829, 6, "1", colours['green'], None),
    RunGroup(888, 1, "2", colours['cyan'], None),
    RunGroup(889, 1, "5", colours['blue'], None),
    # RunGroup(890, 1, "8", colours['grey25'], None),
    RunGroup(891, 1, "10", colours['darkblue'], None),
    RunGroup(702, 6, "No noise", colours['grey25'], None),
]

fig_num = "U"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} bottom half noise, full prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(892, 1, "0.1", colours['darkred'], None),
    RunGroup(893, 1, "0.2", colours['red'], None),
    RunGroup(894, 1, "0.5", colours['orange'], None),
    RunGroup(841, 6, "1", colours['green'], None),
    RunGroup(896, 1, "2", colours['cyan'], None),
    RunGroup(897, 1, "5", colours['blue'], None),
    RunGroup(898, 1, "10", colours['darkblue'], None),
    RunGroup(702, 6, "No noise", colours['grey25'], None),
]

fig_num = "V"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} no noise, bottom half layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(899, 1, "0.1", colours['darkred'], None),
    RunGroup(905, 1, "0.2", colours['red'], None),
    RunGroup(911, 1, "0.5", colours['orange'], None),
    RunGroup(702, 6, "1", colours['green'], None),
    RunGroup(923, 1, "2", colours['cyan'], None),
    RunGroup(929, 1, "5", colours['blue'], None),
    RunGroup(935, 1, "10", colours['darkblue'], None),
]

fig_num = "AC"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} batch size=2000 bottom half noise, bottom half layer 3 prec scan"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(975, 6, "0.1", colours['darkred'], None),
    RunGroup(981, 6, "0.2", colours['red'], None),
    RunGroup(987, 6, "0.5", colours['orange'], None),
    RunGroup(993, 6, "1", colours['green'], None),
    RunGroup(999, 6, "2", colours['cyan'], None),
    RunGroup(1005, 6, "5", colours['blue'], None),
    RunGroup(1011, 6, "10", colours['darkblue'], None),
    RunGroup(1031, 6, "No noise", colours['grey25'], None),
]

fig_num = "AD"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} batch size scan 03"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    RunGroup(946, 1, "10000", colours['red'], None),
    RunGroup(947, 1, "5000", colours['orange'], None),
    RunGroup(948, 1, "2000", colours['green'], None),
    RunGroup(949, 1, "1000", colours['cyan'], None),
    RunGroup(950, 1, "500", colours['blue'], None),
]

fig_num = "AF"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} lr scan with batch size = 2000"
fc["filename"] = f"Fig {fig_num}.svg"
fc["num_datapoints"] = 40
fc["xaxis_range"] = [0, 40]
fc["groups"] = [
    RunGroup(1039, 1, "0.001", colours['red'], None),
    RunGroup(1040, 1, "0.002", colours['darkred'], None),
    RunGroup(1041, 1, "0.005", colours['red'], None),
    RunGroup(1042, 1, "0.006", colours['orange'], None),
    RunGroup(1043, 1, "0.008", colours['green'], None),
    RunGroup(1044, 1, "0.010", colours['cyan'], None),
    RunGroup(1045, 1, "0.015", colours['blue'], None),
    RunGroup(1046, 1, "0.020", colours['darkblue'], None),
]

fig_num = "AG(a)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of noise stddev and precision (coverage=0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (lower image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [67, 88]
fc["groups"] = [
    RunGroup(1266, 6, "0", colours['red'], 0),
    RunGroup(2082, 6, "0", colours['red'], 0.002),
    RunGroup(2088, 6, "0", colours['red'], 0.005),
    RunGroup(2094, 6, "0", colours['red'], 0.01),
    RunGroup(2100, 6, "0", colours['red'], 0.02),
    RunGroup(1506, 6, "0", colours['red'], 0.027),
    RunGroup(1512, 6, "0", colours['red'], 0.038),
    RunGroup(1272, 6, "0", colours['red'], 0.059),
    RunGroup(1278, 6, "0", colours['red'], 0.1),
    RunGroup(1284, 6, "0", colours['red'], 0.2),
    RunGroup(1290, 6, "0", colours['red'], 0.5),
    RunGroup(1296, 6, "0", colours['red'], 0.8),
    RunGroup(1302, 6, "0", colours['red'], 1.0),
    RunGroup(1308, 6, "0", colours['red'], 1.2),
    RunGroup(1314, 6, "1/4", colours['orange'], 0),
    RunGroup(2106, 6, "1/4", colours['orange'], 0.002),
    RunGroup(2112, 6, "1/4", colours['orange'], 0.005),
    RunGroup(2118, 6, "1/4", colours['orange'], 0.01),
    RunGroup(2124, 6, "1/4", colours['orange'], 0.02),
    RunGroup(1518, 6, "1/4", colours['orange'], 0.027),
    RunGroup(1524, 6, "1/4", colours['orange'], 0.038),
    RunGroup(1320, 6, "1/4", colours['orange'], 0.059),
    RunGroup(1326, 6, "1/4", colours['orange'], 0.1),
    RunGroup(1332, 6, "1/4", colours['orange'], 0.2),
    RunGroup(1338, 6, "1/4", colours['orange'], 0.5),
    RunGroup(1344, 6, "1/4", colours['orange'], 0.8),
    RunGroup(1350, 6, "1/4", colours['orange'], 1.0),
    RunGroup(1356, 6, "1/4", colours['orange'], 1.2),
    RunGroup(1362, 6, "1", colours['green'], 0),
    RunGroup(2130, 6, "1", colours['green'], 0.002),
    RunGroup(2136, 6, "1", colours['green'], 0.005),
    RunGroup(2142, 6, "1", colours['green'], 0.01),
    RunGroup(2148, 6, "1", colours['green'], 0.02),
    RunGroup(1530, 6, "1", colours['green'], 0.027),
    RunGroup(1536, 6, "1", colours['green'], 0.038),
    RunGroup(1368, 6, "1", colours['green'], 0.059),
    RunGroup(1374, 6, "1", colours['green'], 0.1),
    RunGroup(1380, 6, "1", colours['green'], 0.2),
    RunGroup(1386, 6, "1", colours['green'], 0.5),
    RunGroup(1392, 6, "1", colours['green'], 0.8),
    RunGroup(1398, 6, "1", colours['green'], 1.0),
    RunGroup(1404, 6, "1", colours['green'], 1.2),
    RunGroup(1410, 6, "4", colours['cyan'], 0),
    RunGroup(2154, 6, "4", colours['cyan'], 0.002),
    RunGroup(2160, 6, "4", colours['cyan'], 0.005),
    RunGroup(2166, 6, "4", colours['cyan'], 0.01),
    RunGroup(2172, 6, "4", colours['cyan'], 0.02),
    RunGroup(1542, 6, "4", colours['cyan'], 0.027),
    RunGroup(1548, 6, "4", colours['cyan'], 0.038),
    RunGroup(1416, 6, "4", colours['cyan'], 0.059),
    RunGroup(1422, 6, "4", colours['cyan'], 0.1),
    RunGroup(1428, 6, "4", colours['cyan'], 0.2),
    RunGroup(1434, 6, "4", colours['cyan'], 0.5),
    RunGroup(1440, 6, "4", colours['cyan'], 0.8),
    RunGroup(1446, 6, "4", colours['cyan'], 1.0),
    RunGroup(1452, 6, "4", colours['cyan'], 1.2),
    RunGroup(1458, 6, "9", colours['blue'], 0),
    RunGroup(2178, 6, "9", colours['blue'], 0.002),
    RunGroup(2184, 6, "9", colours['blue'], 0.005),
    RunGroup(2190, 6, "9", colours['blue'], 0.01),
    RunGroup(2196, 6, "9", colours['blue'], 0.02),
    RunGroup(1554, 6, "9", colours['blue'], 0.027),
    RunGroup(1560, 6, "9", colours['blue'], 0.038),
    RunGroup(1464, 6, "9", colours['blue'], 0.059),
    RunGroup(1470, 6, "9", colours['blue'], 0.1),
    RunGroup(1476, 6, "9", colours['blue'], 0.2),
    RunGroup(1482, 6, "9", colours['blue'], 0.5),
    RunGroup(1488, 6, "9", colours['blue'], 0.8),
    RunGroup(1494, 6, "9", colours['blue'], 1.0),
    RunGroup(1500, 6, "9", colours['blue'], 1.2),
]

fig_num = "AH"
fc = fig_cfgs[fig_num] = fc.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of noise stddev and precision (coverage=0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["yaxis_title"] = "Free energy"
fc["data_field"] = "test/free_e"
fc["xaxis_title"] = "Precision (lower image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [0.0, 0.5]
fc["yaxis_scale"] = [1.0]

fig_num = "AG(c)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of noise stddev and precision (coverage=1.0)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (full image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [52 , 88]
fc["groups"] = [
    RunGroup(1602, 6, "0", colours['red'], 0),
    RunGroup(1962, 6, "0", colours['red'], 0.002),
    RunGroup(1968, 6, "0", colours['red'], 0.005),
    RunGroup(1908, 6, "0", colours['red'], 0.01),
    RunGroup(1902, 6, "0", colours['red'], 0.02),
    RunGroup(1608, 6, "0", colours['red'], 0.027),
    RunGroup(1614, 6, "0", colours['red'], 0.038),
    RunGroup(1620, 6, "0", colours['red'], 0.059),
    RunGroup(1626, 6, "0", colours['red'], 0.1),
    RunGroup(1632, 6, "0", colours['red'], 0.2),
    RunGroup(1638, 6, "0", colours['red'], 0.5),
    RunGroup(1644, 6, "0", colours['red'], 0.8),
    RunGroup(1650, 6, "0", colours['red'], 1.0),
    RunGroup(1656, 6, "0", colours['red'], 1.2),
    RunGroup(1662, 6, "1/4", colours['orange'], 0),
    RunGroup(1974, 6, "1/4", colours['orange'], 0.002),
    RunGroup(1980, 6, "1/4", colours['orange'], 0.005),
    RunGroup(1920, 6, "1/4", colours['orange'], 0.01),
    RunGroup(1914, 6, "1/4", colours['orange'], 0.02),
    RunGroup(1668, 6, "1/4", colours['orange'], 0.027),
    RunGroup(1674, 6, "1/4", colours['orange'], 0.038),
    RunGroup(1680, 6, "1/4", colours['orange'], 0.059),
    RunGroup(1686, 6, "1/4", colours['orange'], 0.1),
    RunGroup(1692, 6, "1/4", colours['orange'], 0.2),
    RunGroup(1698, 6, "1/4", colours['orange'], 0.5),
    RunGroup(1704, 6, "1/4", colours['orange'], 0.8),
    RunGroup(1710, 6, "1/4", colours['orange'], 1.0),
    RunGroup(1716, 6, "1/4", colours['orange'], 1.2),
    RunGroup(1722, 6, "1", colours['green'], 0),
    RunGroup(1986, 6, "1", colours['green'], 0.002),
    RunGroup(1992, 6, "1", colours['green'], 0.005),
    RunGroup(1932, 6, "1", colours['green'], 0.01),
    RunGroup(1926, 6, "1", colours['green'], 0.02),
    RunGroup(1728, 6, "1", colours['green'], 0.027),
    RunGroup(1734, 6, "1", colours['green'], 0.038),
    RunGroup(1740, 6, "1", colours['green'], 0.059),
    RunGroup(1746, 6, "1", colours['green'], 0.1),
    RunGroup(1752, 6, "1", colours['green'], 0.2),
    RunGroup(1758, 6, "1", colours['green'], 0.5),
    RunGroup(1764, 6, "1", colours['green'], 0.8),
    RunGroup(1770, 6, "1", colours['green'], 1.0),
    RunGroup(1776, 6, "1", colours['green'], 1.2),
    RunGroup(1782, 6, "4", colours['cyan'], 0),
    RunGroup(1998, 6, "4", colours['cyan'], 0.002),
    RunGroup(2004, 6, "4", colours['cyan'], 0.005),
    RunGroup(1944, 6, "4", colours['cyan'], 0.01),
    RunGroup(1938, 6, "4", colours['cyan'], 0.02),
    RunGroup(1788, 6, "4", colours['cyan'], 0.027),
    RunGroup(1794, 6, "4", colours['cyan'], 0.038),
    RunGroup(1800, 6, "4", colours['cyan'], 0.059),
    RunGroup(1806, 6, "4", colours['cyan'], 0.1),
    RunGroup(1812, 6, "4", colours['cyan'], 0.2),
    RunGroup(1818, 6, "4", colours['cyan'], 0.5),
    RunGroup(1824, 6, "4", colours['cyan'], 0.8),
    RunGroup(1830, 6, "4", colours['cyan'], 1.0),
    RunGroup(1836, 6, "4", colours['cyan'], 1.2),
    RunGroup(1842, 6, "9", colours['blue'], 0),
    RunGroup(2010, 6, "9", colours['blue'], 0.002),
    RunGroup(2016, 6, "9", colours['blue'], 0.005),
    RunGroup(1956, 6, "9", colours['blue'], 0.01),
    RunGroup(1950, 6, "9", colours['blue'], 0.02),
    RunGroup(1848, 6, "9", colours['blue'], 0.027),
    RunGroup(1854, 6, "9", colours['blue'], 0.038),
    RunGroup(1860, 6, "9", colours['blue'], 0.059),
    RunGroup(1866, 6, "9", colours['blue'], 0.1),
    RunGroup(1872, 6, "9", colours['blue'], 0.2),
    RunGroup(1878, 6, "9", colours['blue'], 0.5),
    RunGroup(1884, 6, "9", colours['blue'], 0.8),
    RunGroup(1890, 6, "9", colours['blue'], 1.0),
    RunGroup(1896, 6, "9", colours['blue'], 1.2),
]

fig_num = "AK"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of noise std and prec (coverage=n1.0/p0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [7, 88]
fc["groups"] = [
    RunGroup(2074, 1, "0", colours['red'], 0),
    RunGroup(2022, 1, "0", colours['red'], 0.002),
    RunGroup(2023, 1, "0", colours['red'], 0.005),
    RunGroup(2024, 1, "0", colours['red'], 0.01),
    RunGroup(2025, 1, "0", colours['red'], 0.02),
    RunGroup(2026, 1, "0", colours['red'], 0.027),
    RunGroup(2075, 1, "0", colours['red'], 0.038),
    RunGroup(2028, 1, "0", colours['red'], 0.059),
    RunGroup(2029, 1, "0", colours['red'], 0.1),
    RunGroup(2030, 1, "0", colours['red'], 0.2),
    RunGroup(2031, 1, "0", colours['red'], 0.5),
    RunGroup(2032, 1, "0", colours['red'], 0.8),
    RunGroup(2033, 1, "0", colours['red'], 1.0),
    RunGroup(2034, 1, "0", colours['red'], 1.2),
    RunGroup(2076, 1, "1.0", colours['green'], 0),
    RunGroup(2035, 1, "1.0", colours['green'], 0.002),
    RunGroup(2036, 1, "1.0", colours['green'], 0.005),
    RunGroup(2037, 1, "1.0", colours['green'], 0.01),
    RunGroup(2038, 1, "1.0", colours['green'], 0.02),
    RunGroup(2039, 1, "1.0", colours['green'], 0.027),
    RunGroup(2077, 1, "1.0", colours['green'], 0.038),
    RunGroup(2041, 1, "1.0", colours['green'], 0.059),
    RunGroup(2042, 1, "1.0", colours['green'], 0.1),
    RunGroup(2043, 1, "1.0", colours['green'], 0.2),
    RunGroup(2044, 1, "1.0", colours['green'], 0.5),
    RunGroup(2045, 1, "1.0", colours['green'], 0.8),
    RunGroup(2046, 1, "1.0", colours['green'], 1.0),
    RunGroup(2047, 1, "1.0", colours['green'], 1.2),
    RunGroup(2078, 1, "2.0", colours['cyan'], 0),
    RunGroup(2048, 1, "2.0", colours['cyan'], 0.002),
    RunGroup(2049, 1, "2.0", colours['cyan'], 0.005),
    RunGroup(2050, 1, "2.0", colours['cyan'], 0.01),
    RunGroup(2051, 1, "2.0", colours['cyan'], 0.02),
    RunGroup(2052, 1, "2.0", colours['cyan'], 0.027),
    RunGroup(2079, 1, "2.0", colours['cyan'], 0.038),
    RunGroup(2054, 1, "2.0", colours['cyan'], 0.059),
    RunGroup(2055, 1, "2.0", colours['cyan'], 0.1),
    RunGroup(2056, 1, "2.0", colours['cyan'], 0.2),
    RunGroup(2057, 1, "2.0", colours['cyan'], 0.5),
    RunGroup(2058, 1, "2.0", colours['cyan'], 0.8),
    RunGroup(2059, 1, "2.0", colours['cyan'], 1.0),
    RunGroup(2060, 1, "2.0", colours['cyan'], 1.2),
    RunGroup(2080, 1, "3.0", colours['blue'], 0),
    RunGroup(2061, 1, "3.0", colours['blue'], 0.002),
    RunGroup(2062, 1, "3.0", colours['blue'], 0.005),
    RunGroup(2063, 1, "3.0", colours['blue'], 0.01),
    RunGroup(2064, 1, "3.0", colours['blue'], 0.02),
    RunGroup(2065, 1, "3.0", colours['blue'], 0.027),
    RunGroup(2081, 1, "3.0", colours['blue'], 0.038),
    RunGroup(2067, 1, "3.0", colours['blue'], 0.059),
    RunGroup(2068, 1, "3.0", colours['blue'], 0.1),
    RunGroup(2069, 1, "3.0", colours['blue'], 0.2),
    RunGroup(2070, 1, "3.0", colours['blue'], 0.5),
    RunGroup(2071, 1, "3.0", colours['blue'], 0.8),
    RunGroup(2072, 1, "3.0", colours['blue'], 1.0),
    RunGroup(2073, 1, "3.0", colours['blue'], 1.2),
]

fig_num = "AL"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} scan of per pixel noise (coverage=0.5/1.0)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Noise range (=2.0 +/- x)"
fc["xaxis_range"] = [0, 3.0]
fc["yaxis_range"] = [72, 88]
fc["groups"] = [
    RunGroup(2202, 6, "0.5", colours['red'], 0),
    RunGroup(2208, 6, "1.0", colours['blue'], 0),
    RunGroup(2214, 6, "0.5", colours['red'], 0.2),
    RunGroup(2220, 6, "1.0", colours['blue'], 0.2),
    RunGroup(2226, 6, "0.5", colours['red'], 0.4),
    RunGroup(2232, 6, "1.0", colours['blue'], 0.4),
    RunGroup(2238, 6, "0.5", colours['red'], 0.6),
    RunGroup(2244, 6, "1.0", colours['blue'], 0.6),
    RunGroup(2250, 6, "0.5", colours['red'], 0.8),
    RunGroup(2256, 6, "1.0", colours['blue'], 0.8),
    RunGroup(2262, 6, "0.5", colours['red'], 1.0),
    RunGroup(2268, 6, "1.0", colours['blue'], 1.0),
    RunGroup(2274, 6, "0.5", colours['red'], 1.2),
    RunGroup(2280, 6, "1.0", colours['blue'], 1.2),
    RunGroup(2286, 6, "0.5", colours['red'], 1.4),
    RunGroup(2292, 6, "1.0", colours['blue'], 1.4),
    RunGroup(2298, 6, "0.5", colours['red'], 1.6),
    RunGroup(2304, 6, "1.0", colours['blue'], 1.6),
    RunGroup(2310, 6, "0.5", colours['red'], 1.8),
    RunGroup(2316, 6, "1.0", colours['blue'], 1.8),
    RunGroup(2322, 6, "0.5", colours['red'], 2.0),
    RunGroup(2328, 6, "1.0", colours['blue'], 2.0),
    RunGroup(2334, 6, "0.5", colours['red'], 2.2),
    RunGroup(2340, 6, "1.0", colours['blue'], 2.2),
    RunGroup(2346, 6, "0.5", colours['red'], 2.4),
    RunGroup(2352, 6, "1.0", colours['blue'], 2.4),
    RunGroup(2358, 6, "0.5", colours['red'], 2.6),
    RunGroup(2364, 6, "1.0", colours['blue'], 2.6),
    RunGroup(2370, 6, "0.5", colours['red'], 2.8),
    RunGroup(2376, 6, "1.0", colours['blue'], 2.8),
    RunGroup(2382, 6, "0.5", colours['red'], 3.0),
    RunGroup(2388, 6, "1.0", colours['blue'], 3.0),
]

fig_num = "AM"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of its mult and prec (noise sd=3.0, cov=1.0)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (full image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [7, 88]
fc["groups"] = [
    RunGroup(2414, 1, "0.5", colours['red'], 0),
    RunGroup(2415, 1, "0.5", colours['red'], 0.002),
    RunGroup(2416, 1, "0.5", colours['red'], 0.005),
    RunGroup(2417, 1, "0.5", colours['red'], 0.01),
    RunGroup(2418, 1, "0.5", colours['red'], 0.02),
    RunGroup(2419, 1, "0.5", colours['red'], 0.027),
    RunGroup(2420, 1, "0.5", colours['red'], 0.038),
    RunGroup(2421, 1, "0.5", colours['red'], 0.059),
    RunGroup(2422, 1, "0.5", colours['red'], 0.1),
    RunGroup(2423, 1, "0.5", colours['red'], 0.2),
    RunGroup(2424, 1, "0.5", colours['red'], 0.5),
    RunGroup(2440, 1, "0.5", colours['red'], 0.8),
    RunGroup(2425, 1, "0.5", colours['red'], 1.0),
    RunGroup(2426, 1, "0.5", colours['red'], 1.2),
    RunGroup(1842, 6, "1.0", colours['blue'], 0),
    RunGroup(2010, 6, "1.0", colours['blue'], 0.002),
    RunGroup(2016, 6, "1.0", colours['blue'], 0.005),
    RunGroup(1956, 6, "1.0", colours['blue'], 0.01),
    RunGroup(1950, 6, "1.0", colours['blue'], 0.02),
    RunGroup(1848, 6, "1.0", colours['blue'], 0.027),
    RunGroup(1854, 6, "1.0", colours['blue'], 0.038),
    RunGroup(1860, 6, "1.0", colours['blue'], 0.059),
    RunGroup(1866, 6, "1.0", colours['blue'], 0.1),
    RunGroup(1872, 6, "1.0", colours['blue'], 0.2),
    RunGroup(1878, 6, "1.0", colours['blue'], 0.5),
    RunGroup(1884, 6, "1.0", colours['blue'], 0.8),
    RunGroup(1890, 6, "1.0", colours['blue'], 1.0),
    RunGroup(1896, 6, "1.0", colours['blue'], 1.2),
    RunGroup(2427, 1, "2.0", colours['green'], 0),
    RunGroup(2428, 1, "2.0", colours['green'], 0.002),
    RunGroup(2429, 1, "2.0", colours['green'], 0.005),
    RunGroup(2430, 1, "2.0", colours['green'], 0.01),
    RunGroup(2431, 1, "2.0", colours['green'], 0.02),
    RunGroup(2432, 1, "2.0", colours['green'], 0.027),
    RunGroup(2433, 1, "2.0", colours['green'], 0.038),
    RunGroup(2434, 1, "2.0", colours['green'], 0.059),
    RunGroup(2435, 1, "2.0", colours['green'], 0.1),
    RunGroup(2436, 1, "2.0", colours['green'], 0.2),
    RunGroup(2437, 1, "2.0", colours['green'], 0.5),
    RunGroup(2441, 1, "2.0", colours['green'], 0.8),
    RunGroup(2438, 1, "2.0", colours['green'], 1.0),
    RunGroup(2439, 1, "2.0", colours['green'], 1.2),
]

fig_num = "AN"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of its mult and prec (noise sd=0)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (full image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [7, 88]
fc["groups"] = [
    RunGroup(2442, 1, "0.5", colours['red'], 0),
    RunGroup(2443, 1, "0.5", colours['red'], 0.002),
    RunGroup(2444, 1, "0.5", colours['red'], 0.005),
    RunGroup(2445, 1, "0.5", colours['red'], 0.01),
    RunGroup(2446, 1, "0.5", colours['red'], 0.02),
    RunGroup(2447, 1, "0.5", colours['red'], 0.027),
    RunGroup(2448, 1, "0.5", colours['red'], 0.038),
    RunGroup(2449, 1, "0.5", colours['red'], 0.059),
    RunGroup(2450, 1, "0.5", colours['red'], 0.1),
    RunGroup(2451, 1, "0.5", colours['red'], 0.2),
    RunGroup(2452, 1, "0.5", colours['red'], 0.5),
    RunGroup(2453, 1, "0.5", colours['red'], 0.8),
    RunGroup(2454, 1, "0.5", colours['red'], 1.0),
    RunGroup(2455, 1, "0.5", colours['red'], 1.2),
    RunGroup(1602, 6, "1.0", colours['blue'], 0),
    RunGroup(1962, 6, "1.0", colours['blue'], 0.002),
    RunGroup(1968, 6, "1.0", colours['blue'], 0.005),
    RunGroup(1908, 6, "1.0", colours['blue'], 0.01),
    RunGroup(1902, 6, "1.0", colours['blue'], 0.02),
    RunGroup(1608, 6, "1.0", colours['blue'], 0.027),
    RunGroup(1614, 6, "1.0", colours['blue'], 0.038),
    RunGroup(1620, 6, "1.0", colours['blue'], 0.059),
    RunGroup(1626, 6, "1.0", colours['blue'], 0.1),
    RunGroup(1632, 6, "1.0", colours['blue'], 0.2),
    RunGroup(1638, 6, "1.0", colours['blue'], 0.5),
    RunGroup(1644, 6, "1.0", colours['blue'], 0.8),
    RunGroup(1650, 6, "1.0", colours['blue'], 1.0),
    RunGroup(1656, 6, "1.0", colours['blue'], 1.2),
    RunGroup(2456, 1, "2.0", colours['green'], 0),
    RunGroup(2457, 1, "2.0", colours['green'], 0.002),
    RunGroup(2458, 1, "2.0", colours['green'], 0.005),
    RunGroup(2459, 1, "2.0", colours['green'], 0.01),
    RunGroup(2460, 1, "2.0", colours['green'], 0.02),
    RunGroup(2461, 1, "2.0", colours['green'], 0.027),
    RunGroup(2462, 1, "2.0", colours['green'], 0.038),
    RunGroup(2463, 1, "2.0", colours['green'], 0.059),
    RunGroup(2464, 1, "2.0", colours['green'], 0.1),
    RunGroup(2465, 1, "2.0", colours['green'], 0.2),
    RunGroup(2466, 1, "2.0", colours['green'], 0.5),
    RunGroup(2467, 1, "2.0", colours['green'], 0.8),
    RunGroup(2468, 1, "2.0", colours['green'], 1.0),
    RunGroup(2469, 1, "2.0", colours['green'], 1.2),
]

fig_num = "AG(b)"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of its mult and prec (noise sd=3, cov=0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (lower image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [67, 88]
fc["groups"] = [
    RunGroup(2470, 1, "x 0.5", colours['red'], 0),
    RunGroup(2471, 1, "x 0.5", colours['red'], 0.002),
    RunGroup(2472, 1, "x 0.5", colours['red'], 0.005),
    RunGroup(2473, 1, "x 0.5", colours['red'], 0.01),
    RunGroup(2474, 1, "x 0.5", colours['red'], 0.02),
    RunGroup(2475, 1, "x 0.5", colours['red'], 0.027),
    RunGroup(2476, 1, "x 0.5", colours['red'], 0.038),
    RunGroup(2477, 1, "x 0.5", colours['red'], 0.059),
    RunGroup(2478, 1, "x 0.5", colours['red'], 0.1),
    RunGroup(2479, 1, "x 0.5", colours['red'], 0.2),
    RunGroup(2480, 1, "x 0.5", colours['red'], 0.5),
    RunGroup(2481, 1, "x 0.5", colours['red'], 0.8),
    RunGroup(2482, 1, "x 0.5", colours['red'], 1.0),
    RunGroup(2483, 1, "x 0.5", colours['red'], 1.2),
    RunGroup(1458, 6, "x 1.0", colours['blue'], 0),
    RunGroup(2178, 6, "x 1.0", colours['blue'], 0.002),
    RunGroup(2184, 6, "x 1.0", colours['blue'], 0.005),
    RunGroup(2190, 6, "x 1.0", colours['blue'], 0.01),
    RunGroup(2196, 6, "x 1.0", colours['blue'], 0.02),
    RunGroup(1554, 6, "x 1.0", colours['blue'], 0.027),
    RunGroup(1560, 6, "x 1.0", colours['blue'], 0.038),
    RunGroup(1464, 6, "x 1.0", colours['blue'], 0.059),
    RunGroup(1470, 6, "x 1.0", colours['blue'], 0.1),
    RunGroup(1477, 6, "x 1.0", colours['blue'], 0.2),
    RunGroup(1482, 6, "x 1.0", colours['blue'], 0.5),
    RunGroup(1488, 6, "x 1.0", colours['blue'], 0.8),
    RunGroup(1494, 6, "x 1.0", colours['blue'], 1.0),
    RunGroup(1500, 6, "x 1.0", colours['blue'], 1.2),
    RunGroup(2484, 1, "x 2.0", colours['green'], 0),
    RunGroup(2485, 1, "x 2.0", colours['green'], 0.002),
    RunGroup(2486, 1, "x 2.0", colours['green'], 0.005),
    RunGroup(2487, 1, "x 2.0", colours['green'], 0.01),
    RunGroup(2488, 1, "x 2.0", colours['green'], 0.02),
    RunGroup(2489, 1, "x 2.0", colours['green'], 0.027),
    RunGroup(2490, 1, "x 2.0", colours['green'], 0.038),
    RunGroup(2491, 1, "x 2.0", colours['green'], 0.059),
    RunGroup(2492, 1, "x 2.0", colours['green'], 0.1),
    RunGroup(2493, 1, "x 2.0", colours['green'], 0.2),
    RunGroup(2494, 1, "x 2.0", colours['green'], 0.5),
    RunGroup(2495, 1, "x 2.0", colours['green'], 0.8),
    RunGroup(2496, 1, "x 2.0", colours['green'], 1.0),
    RunGroup(2497, 1, "x 2.0", colours['green'], 1.2),
]

fig_num = "AQ"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of its mult and prec (noise sd=0, cov=0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (lower image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [7, 88]
fc["groups"] = [
    RunGroup(2498, 1, "0.5", colours['red'], 0),
    RunGroup(2499, 1, "0.5", colours['red'], 0.002),
    RunGroup(2500, 1, "0.5", colours['red'], 0.005),
    RunGroup(2501, 1, "0.5", colours['red'], 0.01),
    RunGroup(2502, 1, "0.5", colours['red'], 0.02),
    RunGroup(2503, 1, "0.5", colours['red'], 0.027),
    RunGroup(2504, 1, "0.5", colours['red'], 0.038),
    RunGroup(2505, 1, "0.5", colours['red'], 0.059),
    RunGroup(2506, 1, "0.5", colours['red'], 0.1),
    RunGroup(2507, 1, "0.5", colours['red'], 0.2),
    RunGroup(2508, 1, "0.5", colours['red'], 0.5),
    RunGroup(2509, 1, "0.5", colours['red'], 0.8),
    RunGroup(2510, 1, "0.5", colours['red'], 1.0),
    RunGroup(2511, 1, "0.5", colours['red'], 1.2),
    RunGroup(1266, 6, "1.0", colours['blue'], 0),
    RunGroup(2082, 6, "1.0", colours['blue'], 0.002),
    RunGroup(2088, 6, "1.0", colours['blue'], 0.005),
    RunGroup(2094, 6, "1.0", colours['blue'], 0.01),
    RunGroup(2100, 6, "1.0", colours['blue'], 0.02),
    RunGroup(1506, 6, "1.0", colours['blue'], 0.027),
    RunGroup(1512, 6, "1.0", colours['blue'], 0.038),
    RunGroup(1272, 6, "1.0", colours['blue'], 0.059),
    RunGroup(1278, 6, "1.0", colours['blue'], 0.1),
    RunGroup(1284, 6, "1.0", colours['blue'], 0.2),
    RunGroup(1290, 6, "1.0", colours['blue'], 0.5),
    RunGroup(1296, 6, "1.0", colours['blue'], 0.8),
    RunGroup(1302, 6, "1.0", colours['blue'], 1.0),
    RunGroup(1308, 6, "1.0", colours['blue'], 1.2),
    RunGroup(2512, 1, "2.0", colours['green'], 0),
    RunGroup(2513, 1, "2.0", colours['green'], 0.002),
    RunGroup(2514, 1, "2.0", colours['green'], 0.005),
    RunGroup(2515, 1, "2.0", colours['green'], 0.01),
    RunGroup(2516, 1, "2.0", colours['green'], 0.02),
    RunGroup(2517, 1, "2.0", colours['green'], 0.027),
    RunGroup(2518, 1, "2.0", colours['green'], 0.038),
    RunGroup(2519, 1, "2.0", colours['green'], 0.059),
    RunGroup(2520, 1, "2.0", colours['green'], 0.1),
    RunGroup(2521, 1, "2.0", colours['green'], 0.2),
    RunGroup(2522, 1, "2.0", colours['green'], 0.5),
    RunGroup(2523, 1, "2.0", colours['green'], 0.8),
    RunGroup(2524, 1, "2.0", colours['green'], 1.0),
    RunGroup(2525, 1, "2.0", colours['green'], 1.2),
]

fig_num = "AR"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_type"] = "2d"
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} 2D scan of its mult and prec B (noise sd=3, cov=0.5)"
fc["filename"] = f"Fig {fig_num}.svg"
fc["xaxis_title"] = "Precision (lower image)"
fc["xaxis_range"] = [0, 1.2]
fc["yaxis_range"] = [7, 88]
fc["groups"] = [
    RunGroup(2530, 1, "1/P", colours['red'], 0),
    RunGroup(2531, 1, "1/P", colours['red'], 0.002),
    RunGroup(2532, 1, "1/P", colours['red'], 0.005),
    RunGroup(2533, 1, "1/P", colours['red'], 0.01),
    RunGroup(2534, 1, "1/P", colours['red'], 0.02),
    RunGroup(2535, 1, "1/P", colours['red'], 0.027),
    RunGroup(2536, 1, "1/P", colours['red'], 0.038),
    RunGroup(2537, 1, "1/P", colours['red'], 0.059),
    RunGroup(2538, 1, "1/P", colours['red'], 0.1),
    RunGroup(2539, 1, "1/P", colours['red'], 0.2),
    RunGroup(2540, 1, "1/P", colours['red'], 0.5),
    RunGroup(2541, 1, "1/P", colours['red'], 0.8),
    RunGroup(2542, 1, "1/P", colours['red'], 1.0),
    RunGroup(2543, 1, "1/P", colours['red'], 1.2),
    RunGroup(1458, 6, "1.0", colours['blue'], 0),
    RunGroup(2178, 6, "1.0", colours['blue'], 0.002),
    RunGroup(2184, 6, "1.0", colours['blue'], 0.005),
    RunGroup(2190, 6, "1.0", colours['blue'], 0.01),
    RunGroup(2196, 6, "1.0", colours['blue'], 0.02),
    RunGroup(1554, 6, "1.0", colours['blue'], 0.027),
    RunGroup(1560, 6, "1.0", colours['blue'], 0.038),
    RunGroup(1464, 6, "1.0", colours['blue'], 0.059),
    RunGroup(1470, 6, "1.0", colours['blue'], 0.1),
    RunGroup(1477, 6, "1.0", colours['blue'], 0.2),
    RunGroup(1482, 6, "1.0", colours['blue'], 0.5),
    RunGroup(1488, 6, "1.0", colours['blue'], 0.8),
    RunGroup(1494, 6, "1.0", colours['blue'], 1.0),
    RunGroup(1500, 6, "1.0", colours['blue'], 1.2),
]

fig_num = "AS"
fc = fig_cfgs[fig_num] = testacc_fig_cfg.copy()
fc["fig_num"] = fig_num
fc["title"] = f"Fig. {fig_num} bottom half noise, bottom half layer 3 prec scan, FashionMNIST"
fc["filename"] = f"Fig {fig_num}.svg"
fc["groups"] = [
    # RunGroup(817, 6, "0.1", colours['darkred'], None),
    RunGroup(2571, 6, "0.2", colours['red'], None),
    RunGroup(2565, 6, "0.5", colours['orange'], None),
    # RunGroup(835, 6, "0.8", colours['green'], None),
    RunGroup(2559, 6, "1", colours['green'], None),
    RunGroup(2577, 1, "2", colours['cyan'], None),
    RunGroup(2578, 1, "5", colours['blue'], None),
    # RunGroup(859, 6, "8", colours['grey25'], None),
    # RunGroup(865, 6, "10", colours['darkblue'], None),
    RunGroup(2547, 6, "No noise", colours['grey25'], None),
]

show_figs = True
save_figs = True
figs = {
    # "D(a)": NickFig(**fig_cfgs["D(a)"], show_fig=show_figs, save_fig=save_figs),
    # "D(b)": NickFig(**fig_cfgs["D(b)"], show_fig=show_figs, save_fig=save_figs),
    # "D(c)": NickFig(**fig_cfgs["D(c)"], show_fig=show_figs, save_fig=save_figs),
    # "G(a)": NickFig(**fig_cfgs["G(a)"], show_fig=show_figs, save_fig=save_figs),
    # "G(b)": NickFig(**fig_cfgs["G(b)"], show_fig=show_figs, save_fig=save_figs),
    # "I(a)": NickFig(**fig_cfgs["I(a)"], show_fig=show_figs, save_fig=save_figs),
    # "I(b)": NickFig(**fig_cfgs["I(b)"], show_fig=show_figs, save_fig=save_figs),
    # "K(a)": NickFig(**fig_cfgs["K(a)"], show_fig=show_figs, save_fig=save_figs),
    # "K(b)": NickFig(**fig_cfgs["K(b)"], show_fig=show_figs, save_fig=save_figs),
    # "L": NickFig(**fig_cfgs["L"], show_fig=show_figs, save_fig=save_figs),
    # "M(a)": NickFig(**fig_cfgs["M(a)"], show_fig=show_figs, save_fig=save_figs),
    # "M(b)": NickFig(**fig_cfgs["M(b)"], show_fig=show_figs, save_fig=save_figs),
    # "M(c)": NickFig(**fig_cfgs["M(c)"], show_fig=show_figs, save_fig=save_figs),
    # "N(a)": NickFig(**fig_cfgs["N(a)"], show_fig=show_figs, save_fig=save_figs),
    # "N(b)": NickFig(**fig_cfgs["N(b)"], show_fig=show_figs, save_fig=save_figs),
    # "P(a)": NickFig(**fig_cfgs["P(a)"], show_fig=show_figs, save_fig=save_figs),
    # "P(b)": NickFig(**fig_cfgs["P(b)"], show_fig=show_figs, save_fig=save_figs),
    # "Q": NickFig(**fig_cfgs["Q"], show_fig=show_figs, save_fig=save_figs),
    # "R": NickFig(**fig_cfgs["R"], show_fig=show_figs, save_fig=save_figs),
    # "S": NickFig(**fig_cfgs["S"], show_fig=show_figs, save_fig=save_figs),
    # "T": NickFig(**fig_cfgs["T"], show_fig=show_figs, save_fig=save_figs),
    # "U": NickFig(**fig_cfgs["U"], show_fig=show_figs, save_fig=save_figs),
    # "V": NickFig(**fig_cfgs["V"], show_fig=show_figs, save_fig=save_figs),
    "AC": NickFig(**fig_cfgs["AC"], show_fig=show_figs, save_fig=save_figs),
    # "AD": NickFig(**fig_cfgs["AD"], show_fig=show_figs, save_fig=save_figs),
    # "AF": NickFig(**fig_cfgs["AF"], show_fig=show_figs, save_fig=save_figs),
    # "AG(a)": NickFig(**fig_cfgs["AG(a)"], show_fig=show_figs, save_fig=save_figs),
    # "AH": NickFig(**fig_cfgs["AH"], show_fig=show_figs, save_fig=save_figs),
    # "AG(c)": NickFig(**fig_cfgs["AG(c)"], show_fig=show_figs, save_fig=save_figs),
    # "AK": NickFig(**fig_cfgs["AK"], show_fig=show_figs, save_fig=save_figs),
    # "AL": NickFig(**fig_cfgs["AL"], show_fig=show_figs, save_fig=save_figs),
    # "AM": NickFig(**fig_cfgs["AM"], show_fig=show_figs, save_fig=save_figs),
    # "AN": NickFig(**fig_cfgs["AN"], show_fig=show_figs, save_fig=save_figs),
    # "AG(b)": NickFig(**fig_cfgs["AG(b)"], show_fig=show_figs, save_fig=save_figs),
    # "AQ": NickFig(**fig_cfgs["AQ"], show_fig=show_figs, save_fig=save_figs),
    # "AR": NickFig(**fig_cfgs["AR"], show_fig=show_figs, save_fig=save_figs),
    # "AS": NickFig(**fig_cfgs["AS"], show_fig=show_figs, save_fig=save_figs),
}

# app = Dash(__name__)
#
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for your data.
#     '''),
#
#     dcc.Graph(
#         id='figD',
#         figure=figs["D"].get_fig()
#     ),
#     dcc.Graph(
#         id='figE',
#         figure=figs["E"].get_fig()
#     ),
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=False)
