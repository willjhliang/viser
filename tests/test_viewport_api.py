"""Focused backend tests for native viewport panes."""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import viser
import viser._client_autobuild
from viser import _messages
from viser._viewport import (
    _encode_image_binary,  # pyright: ignore[reportPrivateImportUsage]
    _validate_image,
)


@pytest.fixture
def viewport_server() -> Generator[viser.ViserServer, None, None]:
    with patch.object(viser._client_autobuild, "ensure_client_is_built", lambda: None):
        server = viser.ViserServer(port=0, verbose=False)
        try:
            yield server
        finally:
            server.stop()


def _messages_in_buffer(server: viser.ViserServer) -> list[_messages.Message]:
    return list(server._websock_server._broadcast_buffer.message_from_id.values())


def _message_ids(
    server: viser.ViserServer, message_type: type[_messages.Message]
) -> list[int]:
    return [
        message_id
        for message_id, message in (
            server._websock_server._broadcast_buffer.message_from_id.items()
        )
        if isinstance(message, message_type)
    ]


def _snapshot(server: viser.ViserServer) -> _messages.ViewportPaneSnapshotMessage:
    snapshots = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneSnapshotMessage)
    ]
    assert len(snapshots) == 1
    return snapshots[0]


def test_viewport_message_lifecycle_and_serialization() -> None:
    assert _messages.ViewportImageMessage.entity_type == "viewport"
    assert _messages.ViewportImageMessage.lifecycle_phase == "create"
    assert _messages.ViewportImageMessage.entity_id_field == "pane_id"
    assert _messages.ViewportImageMessage.include_in_scene_serialization is True

    assert _messages.ViewportPaneUpdateMessage.entity_type == "viewport"
    assert _messages.ViewportPaneUpdateMessage.lifecycle_phase == "update_dict"
    assert _messages.ViewportPaneUpdateMessage.include_in_scene_serialization is True

    assert _messages.ViewportPaneRemoveMessage.entity_type == "viewport"
    assert _messages.ViewportPaneRemoveMessage.lifecycle_phase == "remove"
    assert _messages.ViewportPaneRemoveMessage.include_in_scene_serialization is True

    assert _messages.ViewportPaneSnapshotMessage.include_in_scene_serialization is True


def test_viewport_image_validation() -> None:
    _validate_image(np.zeros((3, 4, 3), dtype=np.uint8))
    _validate_image(np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        _validate_image(np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="dtype"):
        _validate_image(np.zeros((3, 4, 3), dtype=np.bool_))


def test_scene_visibility_uses_persistent_pane_updates(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    assert server.viewport.scene_visible is True

    server.viewport.scene_visible = False
    server.viewport.scene_visible = False
    assert server.viewport.scene_visible is False
    updates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneUpdateMessage)
        and message.pane_id == server.viewport.scene_pane_id
    ]
    assert len(updates) == 1
    assert updates[0].updates == {"visible": False}
    assert _snapshot(server).pane_ids == ()

    server.viewport.scene_visible = True
    assert server.viewport.scene_visible is True
    updates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneUpdateMessage)
        and message.pane_id == server.viewport.scene_pane_id
    ]
    assert len(updates) == 1
    assert updates[0].updates == {"visible": True}


def test_snapshot_tracks_pane_lifecycle_in_broadcast_order(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    assert _snapshot(server).pane_ids == ()

    handle = server.viewport.add_image(
        np.zeros((3, 4, 3), dtype=np.uint8),
        pane_id="camera",
        title="Camera",
    )
    creates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    ]
    assert len(creates) == 1
    assert creates[0].pane_id == handle.pane_id
    assert creates[0].props.title == "Camera"
    assert _snapshot(server).pane_ids == ("camera",)
    assert (
        _message_ids(server, _messages.ViewportImageMessage)[0]
        < _message_ids(server, _messages.ViewportPaneSnapshotMessage)[0]
    )

    handle.title = "Updated"
    handle.fit = "cover"
    handle.visible = False
    handle.image = np.zeros((3, 4, 3), dtype=np.uint8)
    last_frame = np.full((3, 4, 3), 255, dtype=np.uint8)
    handle.image = last_frame
    updates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneUpdateMessage)
    ]
    assert {message.pane_id for message in updates} == {"camera"}
    # Updates coalesce per prop-set, so a late-joining client replays one
    # slot per property group holding the final value. Streamed frames in
    # particular must collapse to only the most recent image.
    updates_by_props = {
        frozenset(message.updates.keys()): message.updates for message in updates
    }
    assert len(updates) == len(updates_by_props)
    assert updates_by_props[frozenset({"title"})]["title"] == "Updated"
    assert updates_by_props[frozenset({"fit"})]["fit"] == "cover"
    assert updates_by_props[frozenset({"visible"})]["visible"] is False
    image_update = updates_by_props[frozenset({"_data", "_format"})]
    expected_format, expected_data = _encode_image_binary(
        last_frame, "auto", jpeg_quality=None
    )
    assert image_update["_format"] == expected_format
    assert image_update["_data"] == expected_data

    handle.remove()
    assert not any(
        isinstance(message, _messages.ViewportPaneUpdateMessage)
        and message.pane_id == handle.pane_id
        for message in _messages_in_buffer(server)
    )
    removes = _message_ids(server, _messages.ViewportPaneRemoveMessage)
    snapshots = _message_ids(server, _messages.ViewportPaneSnapshotMessage)
    assert len(removes) == len(snapshots) == 1
    assert removes[0] < snapshots[0]
    assert _snapshot(server).pane_ids == ()

    with pytest.raises(RuntimeError, match="removed"):
        handle.visible = True


def test_snapshot_includes_hidden_panes_and_excludes_scene(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    server.viewport.add_image(
        np.zeros((2, 2, 3), dtype=np.uint8),
        pane_id="visible",
    )
    server.viewport.add_image(
        np.zeros((2, 2, 3), dtype=np.uint8),
        pane_id="hidden",
        visible=False,
    )

    assert _snapshot(server).pane_ids == ("visible", "hidden")
    assert "scene" not in _snapshot(server).pane_ids

    with pytest.raises(ValueError, match="Unknown or hidden"):
        server.viewport.add_image(
            np.zeros((2, 2, 3), dtype=np.uint8),
            pane_id="relative-to-hidden",
            relative_to="hidden",
        )


def test_explicit_pane_ids_are_unique_and_reusable(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    handle = server.viewport.add_image(frame, pane_id="camera-feed")
    assert handle.pane_id == "camera-feed"

    with pytest.raises(ValueError, match="already exists"):
        server.viewport.add_image(frame, pane_id="camera-feed")
    with pytest.raises(ValueError, match="reserved"):
        server.viewport.add_image(frame, pane_id="scene")
    with pytest.raises(ValueError, match="must not be empty"):
        server.viewport.add_image(frame, pane_id="")
    with pytest.raises(TypeError, match="must be a string"):
        server.viewport.add_image(frame, pane_id=cast(Any, 123))
    with pytest.raises(ValueError, match="Unknown or hidden"):
        server.viewport.add_image(frame, pane_id="bad-relative", relative_to="missing")

    handle.remove()
    replacement = server.viewport.add_image(
        frame,
        pane_id="camera-feed",
        title="Replacement",
    )
    assert replacement.pane_id == "camera-feed"
    assert _snapshot(server).pane_ids == ("camera-feed",)

    creates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
        and message.pane_id == "camera-feed"
    ]
    assert len(creates) == 1
    assert creates[0].props.title == "Replacement"
    assert not any(
        isinstance(message, _messages.ViewportPaneRemoveMessage)
        and message.pane_id == "camera-feed"
        for message in _messages_in_buffer(server)
    )


def test_late_image_update_cannot_mutate_reused_pane_id(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    finish_encode = threading.Event()
    pane_id = "camera-feed"
    old = server.viewport.add_image(
        np.zeros((2, 2, 3), dtype=np.uint8),
        pane_id=pane_id,
    )
    replacement_image = np.full((2, 2, 3), 127, dtype=np.uint8)
    late_image = np.full((2, 2, 3), 255, dtype=np.uint8)
    encode_started = threading.Event()
    errors: list[BaseException] = []

    def delayed_encode(*args: Any, **kwargs: Any) -> tuple[str, bytes]:
        if threading.current_thread().name == "late-viewport-image-update":
            encode_started.set()
            assert finish_encode.wait(timeout=2.0)
        return _encode_image_binary(*args, **kwargs)

    def update_removed_handle() -> None:
        try:
            old.image = late_image
        except BaseException as exc:
            errors.append(exc)

    update_thread: threading.Thread | None = None
    try:
        with patch("viser._viewport._encode_image_binary", side_effect=delayed_encode):
            update_thread = threading.Thread(
                target=update_removed_handle,
                name="late-viewport-image-update",
            )
            update_thread.start()
            assert encode_started.wait(timeout=2.0)

            old.remove()
            replacement = server.viewport.add_image(
                replacement_image,
                pane_id=pane_id,
                title="Replacement",
            )
            finish_encode.set()
            update_thread.join(timeout=2.0)

        assert not update_thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "removed" in str(errors[0])
        assert replacement.title == "Replacement"
        np.testing.assert_array_equal(replacement.image, replacement_image)
        assert not any(
            isinstance(message, _messages.ViewportPaneUpdateMessage)
            and message.pane_id == pane_id
            for message in _messages_in_buffer(server)
        )
    finally:
        finish_encode.set()
        if update_thread is not None:
            update_thread.join(timeout=2.0)


def test_viewport_plotly_message_lifecycle() -> None:
    assert _messages.ViewportPlotlyMessage.entity_type == "viewport"
    assert _messages.ViewportPlotlyMessage.lifecycle_phase == "create"
    assert _messages.ViewportPlotlyMessage.entity_id_field == "pane_id"
    assert _messages.ViewportPlotlyMessage.include_in_scene_serialization is True


def test_plotly_pane_lifecycle_and_updates(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    import json

    server = viewport_server
    handle = server.viewport.add_plotly(
        go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]),
        pane_id="loss-curve",
        title="Loss",
        config={"displayModeBar": False},
    )
    assert handle.pane_id == "loss-curve"
    assert handle.title == "Loss"
    assert handle.visible is True

    creates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPlotlyMessage)
    ]
    assert len(creates) == 1
    assert creates[0].pane_id == "loss-curve"
    assert creates[0].props.title == "Loss"
    plot_dict = json.loads(creates[0].props._plotly_json_str)
    assert plot_dict["config"] == {"displayModeBar": False}
    assert plot_dict["data"][0]["y"] == [0, 1]
    assert _snapshot(server).pane_ids == ("loss-curve",)

    # plotly.min.js must be queued before the pane that needs it.
    run_js_ids = _message_ids(server, _messages.RunJavascriptMessage)
    assert len(run_js_ids) == 1
    assert run_js_ids[0] < _message_ids(server, _messages.ViewportPlotlyMessage)[0]

    handle.figure = go.Figure(data=[go.Scatter(x=[0, 1], y=[1, 0])])
    updates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneUpdateMessage)
        and message.pane_id == "loss-curve"
    ]
    assert len(updates) == 1
    updated_dict = json.loads(updates[0].updates["_plotly_json_str"])
    assert updated_dict["data"][0]["y"] == [1, 0]
    # The config passed at creation applies to updated figures too.
    assert updated_dict["config"] == {"displayModeBar": False}

    handle.remove()
    assert _snapshot(server).pane_ids == ()
    with pytest.raises(RuntimeError, match="removed"):
        handle.figure = go.Figure()
    with pytest.raises(RuntimeError, match="removed"):
        handle.visible = False


def test_plotly_pane_shares_validation_with_image_panes(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")

    server = viewport_server
    figure = go.Figure()
    server.viewport.add_plotly(figure, pane_id="plot")
    with pytest.raises(ValueError, match="already exists"):
        server.viewport.add_plotly(figure, pane_id="plot")
    with pytest.raises(ValueError, match="already exists"):
        server.viewport.add_image(np.zeros((2, 2, 3), dtype=np.uint8), pane_id="plot")
    with pytest.raises(ValueError, match="reserved"):
        server.viewport.add_plotly(figure, pane_id="scene")
    with pytest.raises(ValueError, match="must not be empty"):
        server.viewport.add_plotly(figure, pane_id="")
    with pytest.raises(TypeError, match="must be a string"):
        server.viewport.add_plotly(figure, pane_id=cast(Any, 123))
    with pytest.raises(ValueError, match="Unknown or hidden"):
        server.viewport.add_plotly(figure, relative_to="missing")
    with pytest.raises(ValueError, match="placement"):
        server.viewport.add_plotly(figure, placement=cast(Any, "middle"))

    image_handle = server.viewport.add_image(
        np.zeros((2, 2, 3), dtype=np.uint8), pane_id="camera"
    )
    assert _snapshot(server).pane_ids == ("plot", "camera")
    assert isinstance(image_handle, viser.ViewportImageHandle)


def test_plotly_js_sent_once_across_gui_and_viewport(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")

    server = viewport_server
    server.viewport.add_plotly(go.Figure(), pane_id="a")
    server.viewport.add_plotly(go.Figure(), pane_id="b")
    server.gui.add_plotly(go.Figure())
    run_js = _message_ids(server, _messages.RunJavascriptMessage)
    assert len(run_js) == 1


def test_plotly_pane_theme_template_defaults(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    import json

    import plotly.io as pio

    server = viewport_server

    def pane_create(pane_id: str) -> _messages.ViewportPlotlyMessage:
        creates = [
            message
            for message in _messages_in_buffer(server)
            if isinstance(message, _messages.ViewportPlotlyMessage)
            and message.pane_id == pane_id
        ]
        assert len(creates) == 1
        return creates[0]

    # The stock default template is stripped so the client can substitute a
    # theme-matched template, sent alongside the figure.
    handle = server.viewport.add_plotly(go.Figure(), pane_id="default")
    create = pane_create("default")
    assert "template" not in json.loads(create.props._plotly_json_str)["layout"]
    themes = json.loads(create.props._theme_templates)
    assert themes["light"]["layout"]["plot_bgcolor"] == "white"
    assert (
        themes["dark"]["layout"]["paper_bgcolor"]
        == pio.templates["plotly_dark"].layout.paper_bgcolor
    )

    # Explicitly chosen templates are preserved.
    server.viewport.add_plotly(
        go.Figure(layout=go.Layout(template="plotly_dark")), pane_id="dark"
    )
    dark_layout = json.loads(pane_create("dark").props._plotly_json_str)["layout"]
    assert (
        dark_layout["template"]["layout"]["paper_bgcolor"]
        == pio.templates["plotly_dark"].layout.paper_bgcolor
    )

    # Updates through the handle strip the stock template the same way.
    handle.figure = go.Figure(data=[go.Scatter(x=[0], y=[0])])
    updates = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportPaneUpdateMessage)
        and message.pane_id == "default"
    ]
    assert len(updates) == 1
    updated_layout = json.loads(updates[0].updates["_plotly_json_str"])["layout"]
    assert "template" not in updated_layout


def test_pane_groups_emit_equalize_hints(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")

    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    row = server.viewport.add_row(placement="right", relative_to="scene")
    a = row.add_plotly(go.Figure(), pane_id="a")
    b = row.add_image(frame, pane_id="b")
    c = row.add_plotly(go.Figure(), pane_id="c")
    assert isinstance(a, viser.ViewportPlotlyHandle)
    assert isinstance(b, viser.ViewportImageHandle)
    assert isinstance(c, viser.ViewportPlotlyHandle)

    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(
            message, (_messages.ViewportImageMessage, _messages.ViewportPlotlyMessage)
        )
    }
    # First pane carries the group's placement and no equalize hint; later
    # panes chain off their predecessor and equalize with prior members.
    assert creates["a"].placement == "right"
    assert creates["a"].relative_to == "scene"
    assert creates["a"].equalize_group == ()
    assert creates["b"].placement == "right"
    assert creates["b"].relative_to == "a"
    assert creates["b"].equalize_group == ("a",)
    assert creates["c"].relative_to == "b"
    assert creates["c"].equalize_group == ("a", "b")

    column = server.viewport.add_column(placement="bottom", relative_to="a")
    column.add_image(frame, pane_id="d")
    column.add_image(frame, pane_id="e")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["d"].placement == "bottom"
    assert creates["d"].relative_to == "a"
    assert creates["d"].equalize_group == ()
    assert creates["e"].placement == "bottom"
    assert creates["e"].relative_to == "d"
    assert creates["e"].equalize_group == ("d",)

    # Standalone panes never carry an equalize hint.
    standalone = server.viewport.add_image(frame, pane_id="standalone")
    creates_standalone = [
        message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
        and message.pane_id == "standalone"
    ]
    assert creates_standalone[0].equalize_group == ()
    assert standalone.pane_id == "standalone"

    # Group panes share the registry: IDs still collide across all panes.
    with pytest.raises(ValueError, match="already exists"):
        row.add_plotly(go.Figure(), pane_id="standalone")
    assert _snapshot(server).pane_ids == ("a", "b", "c", "d", "e", "standalone")


def test_pane_groups_skip_hidden_and_removed_members(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    # A hidden member cannot anchor placement or equalize; the group falls
    # back to its own placement when no visible member exists yet.
    column = server.viewport.add_column(placement="bottom", relative_to="scene")
    column.add_image(frame, pane_id="hidden", visible=False)
    column.add_image(frame, pane_id="first-visible")
    second = column.add_image(frame, pane_id="second-visible")

    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["first-visible"].placement == "bottom"
    assert creates["first-visible"].relative_to == "scene"
    assert creates["first-visible"].equalize_group == ()
    assert creates["second-visible"].relative_to == "first-visible"
    assert creates["second-visible"].equalize_group == ("first-visible",)

    # Members hidden or removed after creation are skipped the same way.
    second.visible = False
    third = column.add_image(frame, pane_id="third-visible")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["third-visible"].relative_to == "first-visible"
    assert creates["third-visible"].equalize_group == ("first-visible",)

    # Removing the pane coalesces its buffered create away; the next add
    # anchors on the remaining visible member.
    third.remove()
    column.add_image(frame, pane_id="fourth-visible")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["fourth-visible"].relative_to == "first-visible"
    assert creates["fourth-visible"].equalize_group == ("first-visible",)


def test_image_pane_encoding_validation_and_warnings(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="format"):
        server.viewport.add_image(frame, format=cast(Any, "gif"))
    with pytest.raises(ValueError, match="jpeg_quality"):
        server.viewport.add_image(frame, jpeg_quality=101)
    with pytest.raises(ValueError, match="jpeg_quality"):
        server.viewport.add_image(frame, jpeg_quality=-1)
    # Booleans are ints, but quality=True is always a bug.
    with pytest.raises(ValueError, match="jpeg_quality"):
        server.viewport.add_image(frame, jpeg_quality=cast(Any, True))

    with pytest.warns(UserWarning, match="alpha"):
        handle = server.viewport.add_image(rgba, format="jpeg")
    with pytest.warns(UserWarning, match="alpha"):
        handle.image = rgba

    handle.remove()
    with pytest.warns(UserWarning, match="already removed"):
        handle.remove()


def test_pane_groups_ignore_unrelated_panes_reusing_member_ids(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    row = server.viewport.add_row()
    row.add_image(frame, pane_id="a").remove()
    # A standalone pane reusing the removed member's ID is not group-adopted.
    server.viewport.add_image(frame, pane_id="a")
    row.add_image(frame, pane_id="b")

    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["b"].placement == "right"
    assert creates["b"].relative_to == "scene"
    assert creates["b"].equalize_group == ()

    # A group whose explicit anchor has disappeared falls back to the scene.
    anchor = server.viewport.add_image(frame, pane_id="anchor")
    column = server.viewport.add_column(placement="bottom", relative_to="anchor")
    anchor.remove()
    column.add_image(frame, pane_id="c")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["c"].relative_to == "scene"


def test_pane_grids_fill_row_major_with_equal_cells(
    viewport_server: viser.ViserServer,
) -> None:
    go = pytest.importorskip("plotly.graph_objects")

    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 1"):
        server.viewport.add_grid(0)

    grid = server.viewport.add_grid(2, placement="right", relative_to="scene")
    grid.add_image(frame, pane_id="a")
    grid.add_plotly(go.Figure(), pane_id="b")
    for pane_id in ("c", "d", "e", "f"):
        grid.add_image(frame, pane_id=pane_id)

    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(
            message, (_messages.ViewportImageMessage, _messages.ViewportPlotlyMessage)
        )
    }
    # The top row fills like a row group.
    assert creates["a"].placement == "right"
    assert creates["a"].relative_to == "scene"
    assert creates["a"].equalize_group == ()
    assert creates["b"].placement == "right"
    assert creates["b"].relative_to == "a"
    assert creates["b"].equalize_group == ("a",)
    # Later panes wrap onto new rows beneath the top row. Each equalizes
    # with everything above it in its column, including the top-row pane, so
    # columns stay equally divided.
    assert creates["c"].placement == "bottom"
    assert creates["c"].relative_to == "a"
    assert creates["c"].equalize_group == ("a",)
    assert creates["d"].placement == "bottom"
    assert creates["d"].relative_to == "b"
    assert creates["d"].equalize_group == ("b",)
    assert creates["e"].relative_to == "c"
    assert creates["e"].equalize_group == ("a", "c")
    assert creates["f"].relative_to == "d"
    assert creates["f"].equalize_group == ("b", "d")


def test_pane_grids_skip_dead_cells(
    viewport_server: viser.ViserServer,
) -> None:
    server = viewport_server
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    grid = server.viewport.add_grid(2)
    top_left = grid.add_image(frame, pane_id="a")
    grid.add_image(frame, pane_id="b")
    top_left.remove()

    # The cell below a removed pane falls back like an emptied group: its
    # column's anchor is gone, so it lands beside the scene.
    grid.add_image(frame, pane_id="c")
    grid.add_image(frame, pane_id="d")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["c"].placement == "bottom"
    assert creates["c"].relative_to == "scene"
    assert creates["c"].equalize_group == ()
    # The intact column is unaffected.
    assert creates["d"].placement == "bottom"
    assert creates["d"].relative_to == "b"
    assert creates["d"].equalize_group == ("b",)

    # The next row anchors on the surviving column members.
    grid.add_image(frame, pane_id="e")
    grid.add_image(frame, pane_id="f")
    creates = {
        message.pane_id: message
        for message in _messages_in_buffer(server)
        if isinstance(message, _messages.ViewportImageMessage)
    }
    assert creates["e"].relative_to == "c"
    assert creates["e"].equalize_group == ("c",)
    assert creates["f"].relative_to == "d"
    assert creates["f"].equalize_group == ("b", "d")
