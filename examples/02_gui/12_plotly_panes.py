"""Interactive Plotly viewport panes

Combine the 3D scene with native, resizable 2D Plotly panes. Plots stay fully
interactive (zoom, pan, hover) and dynamically fill their panes, including
when panes are resized in the browser.

**Features demonstrated:**

* :meth:`viser.ViewportApi.add_plotly` for interactive 2D viewport plots
* Dynamic plot sizing that tracks browser-side pane resizing
* Live figure updates through :class:`viser.ViewportPlotlyHandle`
* Plotly config options such as hiding the mode bar
"""

from __future__ import annotations

import math
import time

import numpy as np
import plotly.graph_objects as go
import tyro

import viser


def make_loss_figure(step: int) -> go.Figure:
    """Simulated training curve that grows over time."""

    steps = np.arange(step + 1)
    loss = np.exp(-steps / 40.0) + 0.05 * np.random.default_rng(0).normal(
        size=steps.shape
    )
    figure = go.Figure(
        data=[go.Scatter(x=steps, y=loss, mode="lines", name="loss")],
        layout=go.Layout(
            title="Training loss",
            margin=dict(l=40, r=10, t=40, b=30),
        ),
    )
    return figure


def make_surface_figure() -> go.Figure:
    """Static interactive 3D surface; try dragging to rotate it."""

    x, y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
    z = np.sin(x * 2.0) * np.cos(y * 2.0) * np.exp(-0.3 * (x**2 + y**2))
    return go.Figure(
        data=[go.Surface(z=z, x=x, y=y, showscale=False)],
        layout=go.Layout(title="Interactive surface", margin=dict(l=0, r=0, t=40, b=0)),
    )


def main(update_hz: float = 4.0) -> None:
    if not math.isfinite(update_hz) or update_hz <= 0.0:
        raise ValueError("update_hz must be finite and positive")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10.0, height=10.0)
    server.scene.add_icosphere("/marker", radius=0.35, color=(255, 120, 40))

    loss_pane = server.viewport.add_plotly(
        make_loss_figure(0),
        pane_id="loss-curve",
        title="Loss",
        placement="right",
    )
    server.viewport.add_plotly(
        make_surface_figure(),
        pane_id="surface",
        title="Surface",
        placement="bottom",
        relative_to="loss-curve",
        config={"displayModeBar": False},
    )

    animate = server.gui.add_checkbox("Update loss plot", True)

    step = 0
    while True:
        if animate.value:
            step += 1
            loss_pane.figure = make_loss_figure(step)
        time.sleep(1.0 / update_hz)


if __name__ == "__main__":
    tyro.cli(main)
