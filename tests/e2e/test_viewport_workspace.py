"""End-to-end coverage for native viewport panes and browser-owned layout."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

import viser

from .utils import wait_for_connection


def _wait_for_layout_frame(page: Page) -> None:
    page.evaluate(
        """() => new Promise((resolve) => {
            requestAnimationFrame(() => requestAnimationFrame(resolve));
        })"""
    )


def _wait_for_viewer(page: Page) -> None:
    page.wait_for_function(
        """() => {
            if (document.body.innerText.includes("Connecting...")) return false;
            return window.__viserMutable != null;
        }""",
        timeout=15_000,
    )


def _wait_for_client_count(
    server: viser.ViserServer, expected: int, timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(server.get_clients()) == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"server client count never became {expected}")


def _box(page: Page, selector: str) -> dict[str, float]:
    locator = page.locator(selector)
    locator.evaluate(
        """(element) => Promise.all(
            element.getAnimations().map((animation) =>
                animation.finished.catch(() => undefined)
            )
        )"""
    )
    box = locator.bounding_box()
    assert box is not None
    return {
        "x": box["x"],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"],
    }


def _right(box: dict[str, float]) -> float:
    return box["x"] + box["width"]


def _bottom(box: dict[str, float]) -> float:
    return box["y"] + box["height"]


def _assert_close(actual: float, expected: float, tolerance: float = 1.75) -> None:
    assert abs(actual - expected) <= tolerance, (actual, expected)


def _assert_same_box(
    actual: dict[str, float], expected: dict[str, float], tolerance: float = 1.75
) -> None:
    for key in ("x", "y", "width", "height"):
        _assert_close(actual[key], expected[key], tolerance)


def _assert_horizontal_tiling(
    canvas: dict[str, float],
    left: dict[str, float],
    right: dict[str, float],
) -> None:
    _assert_close(left["x"], canvas["x"])
    _assert_close(left["y"], canvas["y"])
    _assert_close(right["y"], canvas["y"])
    _assert_close(left["height"], canvas["height"])
    _assert_close(right["height"], canvas["height"])
    _assert_close(_right(left), right["x"])
    _assert_close(_right(right), _right(canvas))


def _assert_vertical_tiling(
    canvas: dict[str, float],
    top: dict[str, float],
    bottom: dict[str, float],
) -> None:
    _assert_close(top["x"], canvas["x"])
    _assert_close(top["y"], canvas["y"])
    _assert_close(bottom["x"], canvas["x"])
    _assert_close(top["width"], canvas["width"])
    _assert_close(bottom["width"], canvas["width"])
    _assert_close(_bottom(top), bottom["y"])
    _assert_close(_bottom(bottom), _bottom(canvas))


def _assert_on_grid(value: float, origin: float, cell_size: float) -> None:
    nearest = round((value - origin) / cell_size) * cell_size + origin
    _assert_close(value, nearest)


def _badge_visual_style(page: Page, selector: str) -> dict[str, str]:
    value = page.locator(selector).evaluate(
        """(element) => {
            const style = getComputedStyle(element);
            const properties = [
                "height", "backgroundColor", "color", "fontSize", "fontWeight",
                "lineHeight", "paddingTop", "paddingRight", "paddingBottom",
                "paddingLeft", "borderTopWidth", "borderRightWidth",
                "borderBottomWidth", "borderLeftWidth", "borderTopStyle",
                "borderRightStyle", "borderBottomStyle", "borderLeftStyle",
                "borderTopColor", "borderRightColor", "borderBottomColor",
                "borderLeftColor", "borderTopLeftRadius", "borderTopRightRadius",
                "borderBottomRightRadius", "borderBottomLeftRadius", "boxShadow",
            ];
            return Object.fromEntries(
                properties.map((property) => [property, style[property]])
            );
        }"""
    )
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    assert all(isinstance(item, str) for item in value.values())
    return value


def _computed_background_alpha(page: Page, selector: str) -> int:
    value = page.locator(selector).evaluate(
        """(element) => {
            const style = getComputedStyle(element);
            const canvas = document.createElement("canvas");
            canvas.width = 1;
            canvas.height = 1;
            const context = canvas.getContext("2d");
            if (context === null) return -1;
            context.fillStyle = style.backgroundColor;
            context.fillRect(0, 0, 1, 1);
            return Math.round(
                context.getImageData(0, 0, 1, 1).data[3] *
                Number.parseFloat(style.opacity)
            );
        }"""
    )
    assert isinstance(value, (int, float))
    return int(value)


def _transition_duration_ms(page: Page, selector: str, property_name: str) -> float:
    value = page.locator(selector).evaluate(
        """(element, propertyName) => {
            const style = getComputedStyle(element);
            const properties = style.transitionProperty
                .split(",").map((item) => item.trim());
            const durations = style.transitionDuration
                .split(",").map((item) => {
                    const parsed = Number.parseFloat(item);
                    return item.trim().endsWith("ms") ? parsed : parsed * 1000;
                });
            return properties.reduce((maximum, property, index) => {
                if (property !== "all" && property !== propertyName) return maximum;
                return Math.max(maximum, durations[index % durations.length] ?? 0);
            }, 0);
        }""",
        property_name,
    )
    assert isinstance(value, (int, float))
    return float(value)


def _resolved_css_color(page: Page, color: str) -> str:
    value = page.evaluate(
        """(color) => {
            const probe = document.createElement("span");
            probe.style.color = color;
            document.body.appendChild(probe);
            const resolved = getComputedStyle(probe).color;
            probe.remove();
            return resolved;
        }""",
        color,
    )
    assert isinstance(value, str)
    return value


def _dismiss_software_webgl_notification(page: Page) -> None:
    notification = (
        page.locator(".mantine-Notification-root")
        .filter(has=page.get_by_text("Software WebGL rendering detected", exact=True))
        .first
    )
    try:
        notification.wait_for(state="visible", timeout=1_500)
    except PlaywrightTimeoutError:
        return
    notification.locator(".mantine-Notification-closeButton").click()
    expect(notification).to_have_count(0)


def test_image_pane_content_chrome_and_lifecycle(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Image updates and finalized pane chrome work without layout callbacks."""

    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 0] = 255
    handle = viser_server.viewport.add_image(
        image,
        pane_id="content-image",
        title="Camera feed",
        placement="right",
        format="png",
    )

    pane_selector = f'[data-viewport-pane="{handle.pane_id}"]'
    title_selector = f'[data-viewport-pane-title="{handle.pane_id}"]'
    pane = viser_page.locator(pane_selector)
    title = viser_page.locator(title_selector)
    expect(pane).to_be_visible(timeout=5_000)
    expect(viser_page.locator("[data-viewport-grid-overlay]")).to_have_count(0)
    expect(viser_page.locator("[data-viewport-resize-handle]")).to_have_count(0)

    image_element = pane.locator("img")
    expect(image_element).to_have_attribute("src", re.compile(r"^blob:"))
    initial_src = image_element.get_attribute("src")
    assert initial_src is not None

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    scene_box = _box(viser_page, '[data-viewport-pane="scene"]')
    pane_box = _box(viser_page, pane_selector)
    _assert_horizontal_tiling(canvas, scene_box, pane_box)

    # The compact label is the only grip, is flush with the pane corner, and
    # uses the same opaque, unembellished viser styling as its drag indicator.
    viser_page.mouse.move(
        scene_box["x"] + scene_box["width"] / 2,
        scene_box["y"] + scene_box["height"] / 2,
    )
    expect(title).to_have_css("opacity", "0")
    title_box = _box(viser_page, title_selector)
    assert title_box["width"] < pane_box["width"] * 0.75
    style = _badge_visual_style(viser_page, title_selector)
    assert style["color"] == _resolved_css_color(
        viser_page, "var(--mantine-color-text)"
    )
    assert style["fontWeight"] == "400"
    assert style["boxShadow"] == "none"
    assert _computed_background_alpha(viser_page, title_selector) == 0

    pane_corner = pane.evaluate(
        """(element) => {
            const style = getComputedStyle(element);
            return {
                radius: style.borderTopLeftRadius,
                width: style.borderTopWidth,
                color: style.borderTopColor,
            };
        }"""
    )
    assert style["borderTopLeftRadius"] == pane_corner["radius"]
    assert style["borderTopWidth"] == pane_corner["width"]
    assert style["borderTopColor"] == pane_corner["color"]
    radius_match = re.match(r"^[0-9.]+", style["borderTopLeftRadius"])
    assert radius_match is not None and float(radius_match.group()) > 0
    assert (
        viser_page.locator("[data-viewport-grid-canvas]").evaluate(
            "(element) => getComputedStyle(element).backgroundColor"
        )
        == pane_corner["color"]
    )

    viser_page.mouse.move(
        pane_box["x"] + pane_box["width"] / 2,
        pane_box["y"] + pane_box["height"] / 2,
    )
    expect(title).to_have_css("opacity", "1")
    visible_title = _box(viser_page, title_selector)
    _assert_close(visible_title["x"], pane_box["x"], tolerance=0.25)
    _assert_close(visible_title["y"], pane_box["y"], tolerance=0.25)
    assert _computed_background_alpha(viser_page, title_selector) == 255
    reduced_motion = bool(
        viser_page.evaluate(
            "() => matchMedia('(prefers-reduced-motion: reduce)').matches"
        )
    )
    if not reduced_motion:
        assert _transition_duration_ms(viser_page, title_selector, "opacity") >= 200
        assert _transition_duration_ms(viser_page, pane_selector, "width") > 0

    handle.title = "Updated feed"
    handle.fit = "cover"
    handle.image = np.full((32, 64, 3), 127, dtype=np.uint8)
    expect(title).to_have_text("Updated feed")
    expect(image_element).to_have_css("object-fit", "cover")
    expect(image_element).not_to_have_attribute("src", initial_src)

    # A click focuses the grip, while the rest of the pane's top edge is inert.
    title_box = _box(viser_page, title_selector)
    viser_page.mouse.click(
        title_box["x"] + title_box["width"] / 2,
        title_box["y"] + title_box["height"] / 2,
    )
    expect(title).to_be_focused()
    outside_x = _right(title_box) + 12
    assert outside_x < _right(_box(viser_page, pane_selector)) - 8
    viser_page.mouse.move(outside_x, title_box["y"] + title_box["height"] / 2)
    viser_page.mouse.down()
    viser_page.mouse.move(
        scene_box["x"] + scene_box["width"] / 2,
        scene_box["y"] + scene_box["height"] / 2,
        steps=8,
    )
    expect(viser_page.locator("[data-viewport-drag-indicator]")).to_have_count(0)
    expect(viser_page.locator("[data-viewport-drop-hint]")).to_have_count(0)
    viser_page.mouse.up()

    handle.visible = False
    expect(pane).to_have_count(0)
    _wait_for_layout_frame(viser_page)
    _assert_same_box(
        _box(viser_page, '[data-viewport-pane="scene"]'),
        _box(viser_page, "[data-viewport-grid-canvas]"),
    )
    handle.visible = True
    expect(pane).to_be_visible(timeout=5_000)
    handle.remove()
    expect(pane).to_have_count(0)
    _wait_for_layout_frame(viser_page)
    _assert_same_box(
        _box(viser_page, '[data-viewport-pane="scene"]'),
        _box(viser_page, "[data-viewport-grid-canvas]"),
    )


def test_scene_pane_can_be_hidden_for_2d_only_workspaces(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Three image panes tile the workspace while the scene is hidden."""

    image = np.zeros((9, 16, 3), dtype=np.uint8)
    top = viser_server.viewport.add_image(image, pane_id="top", relative_to="scene")
    viser_server.viewport.add_image(image, pane_id="left", relative_to="scene")
    viser_server.viewport.add_image(image, pane_id="right", relative_to="top")
    viser_server.viewport.scene_visible = False
    assert viser_server.viewport.scene_visible is False

    expect(viser_page.locator('[data-viewport-pane="scene"]')).to_have_count(0)
    pane_boxes = [
        _box(viser_page, f'[data-viewport-pane="{pane_id}"]')
        for pane_id in ("left", "top", "right")
    ]
    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    _assert_close(pane_boxes[0]["x"], canvas["x"])
    _assert_close(_right(pane_boxes[-1]), _right(canvas))
    cell_size = canvas["width"] / 60
    for previous, current in zip(pane_boxes, pane_boxes[1:]):
        _assert_close(_right(previous), current["x"])
        _assert_on_grid(current["x"], canvas["x"], cell_size)
    for pane_box in pane_boxes:
        _assert_close(pane_box["y"], canvas["y"])
        _assert_close(pane_box["height"], canvas["height"])
    pane_widths = [pane_box["width"] for pane_box in pane_boxes]
    assert max(pane_widths) - min(pane_widths) <= cell_size

    top.title = "Top updated"
    expect(viser_page.locator('[data-viewport-pane-title="top"]')).to_have_text(
        "Top updated"
    )
    expect(viser_page.locator('[data-viewport-pane="scene"]')).to_have_count(0)
    viser_server.viewport.scene_visible = True
    expect(viser_page.locator('[data-viewport-pane="scene"]')).to_be_visible()


def test_divider_resizing_snaps_to_square_grid(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Pointer and keyboard resizing snap and clamp without an overlay."""

    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    handle = viser_server.viewport.add_image(
        np.zeros((9, 16, 3), dtype=np.uint8),
        pane_id="resize-image",
        format="png",
    )
    image_selector = f'[data-viewport-pane="{handle.pane_id}"]'
    divider_selector = (
        '[data-viewport-divider=":0"][data-viewport-divider-direction="row"]'
    )
    expect(viser_page.locator(image_selector)).to_be_visible(timeout=5_000)
    divider = viser_page.locator(divider_selector)
    expect(divider).to_have_count(1)

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    scene_before = _box(viser_page, '[data-viewport-pane="scene"]')
    image_before = _box(viser_page, image_selector)
    _assert_horizontal_tiling(canvas, scene_before, image_before)
    cell_size = canvas["width"] / 60

    divider_box = _box(viser_page, divider_selector)
    start_x = divider_box["x"] + divider_box["width"] / 2
    start_y = divider_box["y"] + divider_box["height"] / 2
    viser_page.mouse.move(start_x, start_y)
    viser_page.mouse.down()
    viser_page.mouse.move(start_x + 3.2 * cell_size, start_y, steps=8)
    expect(viser_page.locator("[data-viewport-grid-overlay]")).to_have_count(0)
    expect(viser_page.locator("[data-viewport-drop-hint]")).to_have_count(0)
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)

    scene_after = _box(viser_page, '[data-viewport-pane="scene"]')
    image_after = _box(viser_page, image_selector)
    _assert_horizontal_tiling(canvas, scene_after, image_after)
    delta = scene_after["width"] - scene_before["width"]
    _assert_close(delta, 3 * cell_size)
    _assert_close(image_before["width"] - image_after["width"], delta)
    _assert_on_grid(_right(scene_after), canvas["x"], cell_size)

    divider.focus()
    divider.press("ArrowLeft")
    _wait_for_layout_frame(viser_page)
    scene_after_key = _box(viser_page, '[data-viewport-pane="scene"]')
    _assert_close(scene_after["width"] - scene_after_key["width"], cell_size)
    _assert_on_grid(_right(scene_after_key), canvas["x"], cell_size)

    divider_box = _box(viser_page, divider_selector)
    viser_page.mouse.move(
        divider_box["x"] + divider_box["width"] / 2,
        divider_box["y"] + divider_box["height"] / 2,
    )
    viser_page.mouse.down()
    viser_page.mouse.move(_right(canvas) + 8 * cell_size, start_y, steps=10)
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)
    image_clamped = _box(viser_page, image_selector)
    scene_clamped = _box(viser_page, '[data-viewport-pane="scene"]')
    _assert_horizontal_tiling(canvas, scene_clamped, image_clamped)
    _assert_close(image_clamped["width"], 4 * cell_size)
    _assert_on_grid(_right(scene_clamped), canvas["x"], cell_size)


def test_drag_swap_cancel_and_edge_split(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Drag previews are non-destructive; swap, cancel, and split all work."""

    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    handle = viser_server.viewport.add_image(
        np.zeros((9, 16, 3), dtype=np.uint8),
        pane_id="drag-image",
        title="Drag image",
        format="png",
    )
    image_selector = f'[data-viewport-pane="{handle.pane_id}"]'
    header_selector = f'[data-viewport-pane-header="{handle.pane_id}"]'
    expect(viser_page.locator(image_selector)).to_be_visible(timeout=5_000)
    _dismiss_software_webgl_notification(viser_page)
    assert viser_page.evaluate(
        """() => {
            window.__viewportSceneCanvas = document.querySelector(
                '[data-viewport-pane="scene"] canvas'
            );
            return window.__viewportSceneCanvas !== null;
        }"""
    )

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    scene_before = _box(viser_page, '[data-viewport-pane="scene"]')
    image_before = _box(viser_page, image_selector)
    _assert_horizontal_tiling(canvas, scene_before, image_before)

    header = _box(viser_page, header_selector)
    source_x = header["x"] + header["width"] / 2
    source_y = header["y"] + header["height"] / 2
    target_x = scene_before["x"] + scene_before["width"] / 2
    target_y = scene_before["y"] + scene_before["height"] / 2
    viser_page.mouse.move(source_x, source_y)
    viser_page.mouse.down()
    viser_page.mouse.move(source_x + 8, source_y)
    indicator = viser_page.locator("[data-viewport-drag-indicator]")
    expect(indicator).to_be_visible()
    expect(indicator).to_have_text("Drag image")
    assert _badge_visual_style(
        viser_page, "[data-viewport-drag-indicator]"
    ) == _badge_visual_style(viser_page, header_selector)
    indicator_before = _box(viser_page, "[data-viewport-drag-indicator]")
    viser_page.mouse.move(target_x, target_y, steps=10)
    expect(viser_page.locator('[data-viewport-drop-hint="swap"]')).to_be_visible()
    indicator_after = _box(viser_page, "[data-viewport-drag-indicator]")
    _assert_close(indicator_after["x"] - indicator_before["x"], target_x - source_x - 8)
    _assert_close(indicator_after["y"] - indicator_before["y"], target_y - source_y)
    _assert_same_box(_box(viser_page, image_selector), image_before)
    _assert_same_box(_box(viser_page, '[data-viewport-pane="scene"]'), scene_before)
    viser_page.mouse.up()
    expect(indicator).to_have_count(0)
    _wait_for_layout_frame(viser_page)

    image_swapped = _box(viser_page, image_selector)
    scene_swapped = _box(viser_page, '[data-viewport-pane="scene"]')
    _assert_same_box(image_swapped, scene_before)
    _assert_same_box(scene_swapped, image_before)
    assert viser_page.evaluate(
        """() => window.__viewportSceneCanvas === document.querySelector(
            '[data-viewport-pane="scene"] canvas'
        )"""
    )

    # Escape discards a center-drop preview and leaves both panes untouched.
    header = _box(viser_page, header_selector)
    scene_target = _box(viser_page, '[data-viewport-pane="scene"]')
    viser_page.mouse.move(
        header["x"] + header["width"] / 2,
        header["y"] + header["height"] / 2,
    )
    viser_page.mouse.down()
    viser_page.mouse.move(
        scene_target["x"] + scene_target["width"] / 2,
        scene_target["y"] + scene_target["height"] / 2,
        steps=8,
    )
    expect(viser_page.locator('[data-viewport-drop-hint="swap"]')).to_be_visible()
    viser_page.keyboard.press("Escape")
    expect(viser_page.locator("[data-viewport-drop-hint]")).to_have_count(0)
    expect(indicator).to_have_count(0)
    viser_page.mouse.up()
    _assert_same_box(_box(viser_page, image_selector), image_swapped)
    _assert_same_box(_box(viser_page, '[data-viewport-pane="scene"]'), scene_swapped)

    # A top-edge drop commits a perpendicular split after previewing it.
    header = _box(viser_page, header_selector)
    scene_target = _box(viser_page, '[data-viewport-pane="scene"]')
    viser_page.mouse.move(
        header["x"] + header["width"] / 2,
        header["y"] + header["height"] / 2,
    )
    viser_page.mouse.down()
    viser_page.mouse.move(
        scene_target["x"] + scene_target["width"] / 2,
        scene_target["y"] + scene_target["height"] * 0.1,
        steps=8,
    )
    expect(viser_page.locator('[data-viewport-drop-hint="split"]')).to_be_visible()
    _assert_same_box(_box(viser_page, image_selector), image_swapped)
    _assert_same_box(_box(viser_page, '[data-viewport-pane="scene"]'), scene_swapped)
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)
    _assert_vertical_tiling(
        canvas,
        _box(viser_page, image_selector),
        _box(viser_page, '[data-viewport-pane="scene"]'),
    )
    assert viser_page.evaluate(
        """() => window.__viewportSceneCanvas === document.querySelector(
            '[data-viewport-pane="scene"] canvas'
        )"""
    )


def test_constrained_edge_preview_matches_committed_geometry(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """A short workspace expands its grid and previews final split geometry."""

    viser_page.set_viewport_size({"width": 1280, "height": 80})
    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    _wait_for_layout_frame(viser_page)
    handle = viser_server.viewport.add_image(
        np.zeros((9, 16, 3), dtype=np.uint8),
        pane_id="constrained-preview",
        title="Constrained preview",
        format="png",
    )
    image_selector = f'[data-viewport-pane="{handle.pane_id}"]'
    header_selector = f'[data-viewport-pane-header="{handle.pane_id}"]'
    expect(viser_page.locator(image_selector)).to_be_visible(timeout=5_000)

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    scene_before = _box(viser_page, '[data-viewport-pane="scene"]')
    image_before = _box(viser_page, image_selector)
    _assert_horizontal_tiling(canvas, scene_before, image_before)
    nominal_cell = canvas["width"] / 60
    assert 3 <= canvas["height"] / nominal_cell < 6

    header = _box(viser_page, header_selector)
    viser_page.mouse.move(
        header["x"] + header["width"] / 2,
        header["y"] + header["height"] / 2,
    )
    viser_page.mouse.down()
    viser_page.mouse.move(
        scene_before["x"] + scene_before["width"] / 2,
        scene_before["y"] + scene_before["height"] * 0.1,
        steps=8,
    )
    hint = _box(viser_page, '[data-viewport-drop-hint="split"]')
    _assert_same_box(_box(viser_page, image_selector), image_before)
    _assert_same_box(_box(viser_page, '[data-viewport-pane="scene"]'), scene_before)
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)
    image_after = _box(viser_page, image_selector)
    scene_after = _box(viser_page, '[data-viewport-pane="scene"]')
    _assert_vertical_tiling(canvas, image_after, scene_after)
    _assert_same_box(image_after, hint)


def test_layout_persists_across_same_browser_reload(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """A snapped browser-owned layout survives a page reload."""

    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    handle = viser_server.viewport.add_image(
        np.zeros((9, 16, 3), dtype=np.uint8),
        pane_id="persistent-image",
        format="png",
    )
    image_selector = f'[data-viewport-pane="{handle.pane_id}"]'
    divider_selector = (
        '[data-viewport-divider=":0"][data-viewport-divider-direction="row"]'
    )
    expect(viser_page.locator(image_selector)).to_be_visible(timeout=5_000)
    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    cell_size = canvas["width"] / 60
    divider = _box(viser_page, divider_selector)
    start_x = divider["x"] + divider["width"] / 2
    start_y = divider["y"] + divider["height"] / 2
    viser_page.mouse.move(start_x, start_y)
    viser_page.mouse.down()
    viser_page.mouse.move(start_x + 4.2 * cell_size, start_y, steps=8)
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)

    scene_before = _box(viser_page, '[data-viewport-pane="scene"]')
    image_before = _box(viser_page, image_selector)
    _assert_on_grid(_right(scene_before), canvas["x"], cell_size)
    viser_page.wait_for_function(
        """() => Object.keys(localStorage).some(
            (key) => key.startsWith("viser.viewport.layout.v1:")
        )"""
    )

    viser_page.reload()
    _wait_for_viewer(viser_page)
    expect(viser_page.locator(image_selector)).to_be_visible(timeout=5_000)
    _assert_same_box(_box(viser_page, '[data-viewport-pane="scene"]'), scene_before)
    _assert_same_box(_box(viser_page, image_selector), image_before)


def test_snapshot_reconciles_layout_after_disconnect(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Reconnect retains stable placement, prunes removal, and inserts new panes."""

    viser_server.gui.configure_theme(control_layout="fixed", show_logo=False)
    stable = viser_server.viewport.add_image(
        np.zeros((8, 8, 3), dtype=np.uint8),
        pane_id="stable-pane",
        title="Stable",
        format="png",
    )
    removed = viser_server.viewport.add_image(
        np.ones((8, 8, 3), dtype=np.uint8),
        pane_id="removed-pane",
        title="Removed",
        placement="right",
        relative_to=stable.pane_id,
        format="png",
    )
    stable_selector = f'[data-viewport-pane="{stable.pane_id}"]'
    removed_selector = f'[data-viewport-pane="{removed.pane_id}"]'
    expect(viser_page.locator(removed_selector)).to_be_visible(timeout=5_000)
    _dismiss_software_webgl_notification(viser_page)

    # Put the stable pane left of the scene; this differs from declaration order
    # and is therefore an observable persisted placement after reconnect.
    stable_header = _box(viser_page, f'[data-viewport-pane-header="{stable.pane_id}"]')
    scene = _box(viser_page, '[data-viewport-pane="scene"]')
    viser_page.mouse.move(
        stable_header["x"] + stable_header["width"] / 2,
        stable_header["y"] + stable_header["height"] / 2,
    )
    viser_page.mouse.down()
    viser_page.mouse.move(
        stable_header["x"] + stable_header["width"] / 2 + 8,
        stable_header["y"] + stable_header["height"] / 2,
    )
    viser_page.mouse.move(
        scene["x"] + scene["width"] / 2,
        scene["y"] + scene["height"] / 2,
        steps=8,
    )
    expect(viser_page.locator('[data-viewport-drop-hint="swap"]')).to_be_visible()
    viser_page.mouse.up()
    _wait_for_layout_frame(viser_page)
    assert (
        _box(viser_page, stable_selector)["x"]
        < _box(viser_page, '[data-viewport-pane="scene"]')["x"]
    )

    viser_page.goto("about:blank")
    _wait_for_client_count(viser_server, 0)
    removed.remove()
    added = viser_server.viewport.add_image(
        np.full((8, 8, 3), 127, dtype=np.uint8),
        pane_id="new-pane",
        title="New",
        placement="right",
        relative_to="scene",
        format="png",
    )

    wait_for_connection(viser_page, viser_server.get_port())
    added_selector = f'[data-viewport-pane="{added.pane_id}"]'
    expect(viser_page.locator(stable_selector)).to_be_visible(timeout=5_000)
    expect(viser_page.locator(removed_selector)).to_have_count(0)
    expect(viser_page.locator(added_selector)).to_be_visible(timeout=5_000)
    stable_box = _box(viser_page, stable_selector)
    scene_box = _box(viser_page, '[data-viewport-pane="scene"]')
    added_box = _box(viser_page, added_selector)
    assert stable_box["x"] < scene_box["x"] < added_box["x"]

    persisted_ids = viser_page.evaluate(
        """() => {
            const key = Object.keys(localStorage).find(
                (candidate) => candidate.startsWith("viser.viewport.layout.v1:")
            );
            if (key === undefined) return [];
            const collect = (node) => node.type === "pane"
                ? [node.pane_id]
                : node.children.flatMap(collect);
            return collect(JSON.parse(localStorage.getItem(key)).root);
        }"""
    )
    assert persisted_ids == [stable.pane_id, "scene", added.pane_id]


def test_image_pane_renders_in_standalone_html(
    page: Page, viser_server: viser.ViserServer
) -> None:
    """Image panes and their snapshot survive serializer/embed playback."""

    handle = viser_server.viewport.add_image(
        np.full((16, 24, 3), 200, dtype=np.uint8),
        pane_id="__proto__",
        title="Embedded image",
        placement="bottom",
        format="png",
    )
    html = viser_server.scene.as_html()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as file:
        file.write(html)
        path = Path(file.name)

    try:
        page.goto(path.as_uri())
        pane = page.locator(f'[data-viewport-pane="{handle.pane_id}"]')
        expect(pane).to_be_visible(timeout=15_000)
        expect(
            page.locator(f'[data-viewport-pane-header="{handle.pane_id}"]')
        ).to_have_text("Embedded image")
        expect(pane.locator("img")).to_have_attribute("src", re.compile(r"^blob:"))
        expect(page.locator("[data-viewport-divider]")).to_have_count(1)
        expect(page.locator("[data-viewport-grid-overlay]")).to_have_count(0)
    finally:
        path.unlink(missing_ok=True)


def test_recording_rewind_resets_later_viewport_panes(
    page: Page, viser_server: viser.ViserServer
) -> None:
    """Seeking before a pane's creation removes temporal viewport state."""

    serializer = viser_server.get_scene_serializer()
    serializer.insert_sleep(0.3)
    handle = viser_server.viewport.add_image(
        np.full((12, 18, 3), 80, dtype=np.uint8),
        title="Later pane",
        format="png",
    )
    serializer.insert_sleep(10.0)
    html = serializer.as_html()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as file:
        file.write(html)
        path = Path(file.name)

    try:
        page.goto(path.as_uri())
        pane = page.locator(f'[data-viewport-pane="{handle.pane_id}"]')
        expect(pane).to_be_visible(timeout=15_000)
        time_input = page.get_by_role("textbox").first
        expect(time_input).to_be_visible(timeout=5_000)
        time_input.fill("0")
        expect(pane).to_have_count(0, timeout=5_000)
        expect(page.locator('[data-viewport-pane="scene"]')).to_be_visible()
        expect(page.locator('[data-viewport-pane-header="scene"]')).to_have_count(0)
    finally:
        path.unlink(missing_ok=True)


def test_floating_gui_stays_above_viewport_chrome(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Viewport dividers must not paint over or intercept the floating GUI."""

    viser_server.gui.configure_theme(control_layout="floating", show_logo=False)
    handle = viser_server.viewport.add_image(
        np.zeros((8, 8, 3), dtype=np.uint8),
        pane_id="stacking-pane",
        format="png",
    )
    expect(
        viser_page.locator(f'[data-viewport-pane="{handle.pane_id}"]')
    ).to_be_visible(timeout=5_000)

    divider = viser_page.locator("[data-viewport-divider]").first
    floating = viser_page.locator("[data-floating-window]").first
    divider_box = divider.bounding_box()
    assert divider_box is not None
    expect(floating).to_be_visible()

    probe_x = divider_box["x"] + divider_box["width"] / 2
    probe_y = 100
    floating_is_on_top = floating.evaluate(
        """(element, probe) => {
            element.style.left = (probe.x - 80) + "px";
            element.style.top = "40px";
            const top = document.elementFromPoint(probe.x, probe.y);
            return top?.closest("[data-floating-window]") === element;
        }""",
        {"x": probe_x, "y": probe_y},
    )
    assert floating_is_on_top


def test_plotly_pane_is_interactive_and_fills_pane_across_resizes(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """A Plotly pane renders interactively and always fills its pane."""

    import plotly.graph_objects as go

    handle = viser_server.viewport.add_plotly(
        go.Figure(data=[go.Scatter(x=[0, 1, 2, 3], y=[0, 1, 4, 9], mode="lines")]),
        pane_id="plot",
        title="Plot",
    )

    pane_selector = '[data-viewport-pane="plot"]'
    plot_selector = f"{pane_selector} .js-plotly-plot"
    expect(viser_page.locator(pane_selector)).to_be_visible(timeout=10_000)
    expect(viser_page.locator(plot_selector)).to_be_visible(timeout=10_000)

    def wait_for_plot_to_fill_pane() -> None:
        viser_page.wait_for_function(
            """() => {
                const pane = document.querySelector(
                    '[data-viewport-pane-content="plot"]');
                const plot = pane?.querySelector('.js-plotly-plot');
                if (!pane || !plot) return false;
                const paneRect = pane.getBoundingClientRect();
                const plotRect = plot.getBoundingClientRect();
                return (
                    paneRect.width > 0 &&
                    paneRect.height > 0 &&
                    Math.abs(paneRect.width - plotRect.width) < 2.5 &&
                    Math.abs(paneRect.height - plotRect.height) < 2.5
                );
            }""",
            timeout=10_000,
        )

    def x_axis_range() -> list[float]:
        return viser_page.eval_on_selector(
            plot_selector,
            "(plot) => plot._fullLayout.xaxis.range.map(Number)",
        )

    wait_for_plot_to_fill_pane()

    # Live figure updates from the server appear without a reload.
    updated = go.Figure(
        data=[
            go.Scatter(x=[0, 1, 2, 3], y=[0, 1, 4, 9], mode="lines"),
            go.Scatter(x=[0, 1, 2, 3], y=[9, 4, 1, 0], mode="lines"),
        ]
    )
    handle.figure = updated
    viser_page.wait_for_function(
        f"""() => document.querySelectorAll(
            '{plot_selector} .scatterlayer .trace').length === 2""",
        timeout=10_000,
    )

    # Drag-zoom inside the plot area: the x-axis range must narrow, proving
    # the pane hosts a live Plotly instance rather than a static image.
    initial_range = x_axis_range()
    plot_box = _box(viser_page, plot_selector)
    center_x = plot_box["x"] + plot_box["width"] / 2
    center_y = plot_box["y"] + plot_box["height"] / 2
    viser_page.mouse.move(center_x - plot_box["width"] / 5, center_y)
    viser_page.mouse.down()
    viser_page.mouse.move(center_x + plot_box["width"] / 5, center_y, steps=8)
    viser_page.mouse.up()
    viser_page.wait_for_function(
        f"""() => {{
            const plot = document.querySelector('{plot_selector}');
            const range = plot?._fullLayout?.xaxis?.range;
            if (!range) return false;
            return range[1] - range[0] < {initial_range[1] - initial_range[0]} * 0.9;
        }}""",
        timeout=10_000,
    )
    zoomed_range = x_axis_range()

    # Shrinking the browser window shrinks the pane; the plot must track it
    # while preserving interaction state (zoom) through uirevision.
    viser_page.set_viewport_size({"width": 640, "height": 420})
    _wait_for_layout_frame(viser_page)
    wait_for_plot_to_fill_pane()
    assert x_axis_range() == zoomed_range

    viser_page.set_viewport_size({"width": 960, "height": 600})
    _wait_for_layout_frame(viser_page)
    wait_for_plot_to_fill_pane()


def test_plotly_pane_template_tracks_viser_theme(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """Untemplated figures follow viser's light/dark theme live."""

    import plotly.graph_objects as go

    viser_server.viewport.add_plotly(
        go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]),
        pane_id="plot",
        title="Plot",
    )
    templated = viser_server.viewport.add_plotly(
        go.Figure(layout=go.Layout(template="plotly")),
        pane_id="templated",
        title="Templated",
    )

    def wait_for_paper_bgcolor(pane_id: str, expected: str) -> None:
        viser_page.wait_for_function(
            f"""() => {{
                const plot = document.querySelector(
                    '[data-viewport-pane="{pane_id}"] .js-plotly-plot');
                return plot?._fullLayout?.paper_bgcolor === '{expected}';
            }}""",
            timeout=10_000,
        )

    # plotly_white in light mode, plotly_dark in dark mode, switched live.
    wait_for_paper_bgcolor("plot", "white")
    viser_server.gui.configure_theme(dark_mode=True)
    wait_for_paper_bgcolor("plot", "rgb(17,17,17)")
    viser_server.gui.configure_theme(dark_mode=False)
    wait_for_paper_bgcolor("plot", "white")

    # An explicitly templated figure keeps its template in both modes. The
    # stock "plotly" template is indistinguishable from an untouched figure,
    # so it is themed like one; any other explicit template is preserved.
    templated.figure = go.Figure(layout=go.Layout(template="seaborn"))
    wait_for_paper_bgcolor("templated", "white")
    viser_server.gui.configure_theme(dark_mode=True)
    wait_for_paper_bgcolor("templated", "white")


def test_pane_group_divides_space_into_exact_thirds(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """add_row panes re-equalize on each insertion, landing on grid lines."""

    import plotly.graph_objects as go

    row = viser_server.viewport.add_row()
    for name in ("a", "b", "c"):
        row.add_plotly(
            go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]),
            pane_id=name,
            title=name,
        )
    viser_server.viewport.scene_visible = False

    expect(viser_page.locator('[data-viewport-pane="c"]')).to_be_visible(timeout=10_000)
    expect(viser_page.locator('[data-viewport-pane="scene"]')).to_have_count(0)

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    cell_size = canvas["width"] / 60
    boxes = [
        _box(viser_page, f'[data-viewport-pane="{name}"]') for name in ("a", "b", "c")
    ]
    _assert_close(boxes[0]["x"], canvas["x"])
    _assert_close(_right(boxes[2]), _right(canvas))
    for previous, current in zip(boxes, boxes[1:]):
        _assert_close(_right(previous), current["x"])
    for box in boxes:
        # Thirds land exactly on the 60-column grid: 20 cells each.
        _assert_close(box["width"], 20 * cell_size)
        _assert_on_grid(box["x"], canvas["x"], cell_size)


def test_pane_grid_divides_space_into_equal_cells(
    viser_page: Page, viser_server: viser.ViserServer
) -> None:
    """add_grid fills row-major with equal columns and rows."""

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    names = [f"cell-{row}{col}" for row in range(3) for col in range(3)]
    grid = viser_server.viewport.add_grid(3)
    for name in names:
        grid.add_image(frame, pane_id=name, title=name)
    viser_server.viewport.scene_visible = False

    expect(viser_page.locator('[data-viewport-pane="cell-22"]')).to_be_visible(
        timeout=10_000
    )
    expect(viser_page.locator('[data-viewport-pane="scene"]')).to_have_count(0)

    canvas = _box(viser_page, "[data-viewport-grid-canvas]")
    cell_size = canvas["width"] / 60
    boxes = [
        [
            _box(viser_page, f'[data-viewport-pane="cell-{row}{col}"]')
            for col in range(3)
        ]
        for row in range(3)
    ]

    # Columns are exact thirds of the workspace width: 20 grid cells each.
    for row in boxes:
        _assert_close(row[0]["x"], canvas["x"])
        _assert_close(_right(row[2]), _right(canvas))
        for left, right in zip(row, row[1:]):
            _assert_close(_right(left), right["x"])
        for box in row:
            _assert_close(box["width"], 20 * cell_size)
            _assert_on_grid(box["x"], canvas["x"], cell_size)
            _assert_on_grid(box["y"], canvas["y"], cell_size)
            _assert_close(box["height"], row[0]["height"])

    # Rows tile the full height and are equal to within one grid cell; row
    # heights snap to the square grid, which may not divide into exact thirds.
    _assert_close(boxes[0][0]["y"], canvas["y"])
    _assert_close(_bottom(boxes[2][0]), _bottom(canvas))
    for upper, lower in zip(boxes, boxes[1:]):
        for upper_box, lower_box in zip(upper, lower):
            _assert_close(_bottom(upper_box), lower_box["y"])
    for row in boxes:
        _assert_close(
            row[0]["height"], canvas["height"] / 3, tolerance=cell_size + 1.75
        )
