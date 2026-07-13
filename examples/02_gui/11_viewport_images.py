"""Image viewport panes

Combine the 3D scene with a native, resizable 2D image pane. Existing GUI
controls can update either pane, and the image handle can be refreshed at
interactive rates for video-like data.

**Features demonstrated:**

* :meth:`viser.ViewportApi.add_image` for native 2D viewport content
* Resizable viewport splitting and browser-managed pane arrangement
* Browser-local layout persistence through a stable pane ID
* GUI controls shared across 2D and 3D panes
* Streaming image updates through :class:`viser.ViewportImageHandle`
"""

from __future__ import annotations

import math
import time

import numpy as np
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


def main(fps: float = 20.0) -> None:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=10.0, height=10.0)
    marker = server.scene.add_icosphere("/marker", radius=0.35, color=(255, 120, 40))

    image_pane = server.viewport.add_image(
        make_frame(0.0),
        pane_id="camera-feed",
        title="2D stream",
        placement="right",
        fit="contain",
        format="jpeg",
        jpeg_quality=85,
    )

    animate = server.gui.add_checkbox("Animate image", True)
    fit = server.gui.add_dropdown(
        "Image fit", ("contain", "cover", "fill"), initial_value="contain"
    )
    marker_x = server.gui.add_slider("3D marker X", -3.0, 3.0, 0.01, 0.0)

    @fit.on_update
    def _(_) -> None:
        image_pane.fit = fit.value

    @marker_x.on_update
    def _(_) -> None:
        marker.position = (marker_x.value, 0.0, 0.35)

    phase = 0.0
    frame_time = 1.0 / fps
    next_frame_time = time.monotonic()
    while True:
        if animate.value:
            phase = (phase + frame_time * 0.2) % 1.0
            image_pane.image = make_frame(phase)
        next_frame_time += frame_time
        time.sleep(max(0.0, next_frame_time - time.monotonic()))

        # Avoid a catch-up loop if encoding or a suspended process falls behind.
        next_frame_time = max(next_frame_time, time.monotonic() - frame_time)


if __name__ == "__main__":
    tyro.cli(main)
