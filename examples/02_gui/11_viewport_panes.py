"""Viewport panes

Combine the 3D scene with native, resizable 2D panes: a streamed image, an
interactive line plot, and an interactive 3D plot. Panes added through
:meth:`viser.ViewportApi.add_column` (or ``add_row``) divide their space
equally — three panes make exact thirds — and Plotly panes stay fully
interactive (zoom, pan, hover) while dynamically filling their pane, even as
panes are resized in the browser.

**Features demonstrated:**

* :meth:`viser.ViewportApi.add_image` for streamed 2D viewport content
* :meth:`viser.ViewportApi.add_plotly` for interactive 2D and 3D plots
* :meth:`viser.ViewportApi.add_column` for equally divided pane groups
* Live updates through :class:`viser.ViewportImageHandle` and
  :class:`viser.ViewportPlotlyHandle`
* Browser-local layout persistence through stable pane IDs
"""

from __future__ import annotations

import math
import time

import numpy as np
import plotly.graph_objects as go
import tyro

import viser


def make_frame(phase: float, height: int = 360, width: int = 640) -> np.ndarray:
    """Generate a moving RGB gradient without external image dependencies."""

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    red = np.broadcast_to((x + phase) % 1.0, (height, width))
    green = np.broadcast_to((y + phase * 0.5) % 1.0, (height, width))
    blue = (x + y + phase * 0.25) % 1.0
    return np.stack((red, green, blue), axis=-1)


def make_loss_figure(step: int) -> go.Figure:
    """Simulated training curve that grows over time."""

    steps = np.arange(step + 1)
    loss = np.exp(-steps / 40.0) + 0.05 * np.random.default_rng(0).normal(
        size=steps.shape
    )
    return go.Figure(
        data=[go.Scatter(x=steps, y=loss, mode="lines")],
        layout=go.Layout(margin=dict(l=40, r=10, t=20, b=30)),
    )


def make_surface_figure() -> go.Figure:
    """Static interactive 3D surface; try dragging to rotate it."""

    x, y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
    z = np.sin(x * 2.0) * np.cos(y * 2.0) * np.exp(-0.3 * (x**2 + y**2))
    return go.Figure(
        data=[go.Surface(z=z, x=x, y=y, showscale=False)],
        layout=go.Layout(margin=dict(l=0, r=0, t=0, b=0)),
    )


def main(fps: float = 20.0, plot_hz: float = 4.0) -> None:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(plot_hz) or plot_hz <= 0.0:
        raise ValueError("plot_hz must be finite and positive")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10.0, height=10.0)
    server.scene.add_icosphere("/marker", radius=0.35, color=(255, 120, 40))

    # An image stream and two plots stacked beside the scene, kept equal
    # heights (exact thirds).
    column = server.viewport.add_column(placement="right", relative_to="scene")
    image_pane = column.add_image(
        make_frame(0.0),
        pane_id="camera-feed",
        title="2D stream",
        fit="contain",
        format="jpeg",
        jpeg_quality=85,
    )
    loss_pane = column.add_plotly(make_loss_figure(0), pane_id="loss", title="Loss")
    column.add_plotly(
        make_surface_figure(),
        pane_id="surface",
        title="Surface",
        config={"displayModeBar": False},
    )

    animate = server.gui.add_checkbox("Animate panes", True)
    fit = server.gui.add_dropdown(
        "Image fit", ("contain", "cover", "fill"), initial_value="contain"
    )

    @fit.on_update
    def _(_) -> None:
        image_pane.fit = fit.value

    phase = 0.0
    step = 0
    frame_time = 1.0 / fps
    plot_every = max(1, round(fps / plot_hz))
    next_frame_time = time.monotonic()
    while True:
        if animate.value:
            phase = (phase + frame_time * 0.2) % 1.0
            image_pane.image = make_frame(phase)
            step += 1
            if step % plot_every == 0:
                loss_pane.figure = make_loss_figure(step)
        next_frame_time += frame_time
        time.sleep(max(0.0, next_frame_time - time.monotonic()))

        # Avoid a catch-up loop if encoding or a suspended process falls behind.
        next_frame_time = max(next_frame_time, time.monotonic() - frame_time)


if __name__ == "__main__":
    tyro.cli(main)
