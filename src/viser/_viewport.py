"""Python API for native viewport panes."""

from __future__ import annotations

import copy
import dataclasses
import threading
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Literal, cast

import numpy as np
from typing_extensions import TypeAlias

from . import _messages
from ._scene_api import _encode_image_binary

if TYPE_CHECKING:
    from ._viser import ViserServer


ViewportPaneFit: TypeAlias = Literal["contain", "cover", "fill"]
ViewportPanePlacement: TypeAlias = Literal["left", "right", "top", "bottom"]


@dataclasses.dataclass
class _ViewportImageHandleState:
    pane_id: str
    props: _messages.ViewportImageProps
    api: ViewportApi
    image: np.ndarray
    requested_format: Literal["auto", "jpeg", "png"]
    jpeg_quality: int | None
    removed: bool = False


class ViewportImageHandle:
    """Handle for updating or removing a native viewport image pane."""

    def __init__(self, state: _ViewportImageHandleState) -> None:
        self._impl = state

    def _check_not_removed(self) -> None:
        if self._impl.removed:
            raise RuntimeError(f"Cannot update a removed {type(self).__name__}.")

    def _queue_update(self, updates: Dict[str, Any]) -> None:
        self._impl.api._websock_interface.queue_message(
            _messages.ViewportPaneUpdateMessage(
                pane_id=self._impl.pane_id,
                updates=updates,
            )
        )

    @property
    def pane_id(self) -> str:
        """Stable identifier used to restore browser-managed layouts."""

        return self._impl.pane_id

    @property
    def image(self) -> np.ndarray:
        """Current image. Assign a new array to stream another frame."""

        return self._impl.image

    @image.setter
    def image(self, image: np.ndarray) -> None:
        self._check_not_removed()
        image = _validate_image(image)
        resolved_format, data = _encode_image_binary(
            image,
            self._impl.requested_format,
            jpeg_quality=self._impl.jpeg_quality,
        )
        if self._impl.requested_format == "jpeg" and image.shape[2] == 4:
            warnings.warn(
                "Encoding an RGBA viewport image as JPEG discards its alpha channel.",
                stacklevel=2,
            )
        with self._impl.api._lock:
            # Encoding can be expensive. Recheck after taking the lock so a
            # concurrent remove cannot queue an update for a reused pane ID.
            self._check_not_removed()
            self._impl.image = image.copy()
            self._impl.props._format = resolved_format
            self._impl.props._data = data
            self._queue_update({"_format": resolved_format, "_data": data})

    @property
    def title(self) -> str:
        """Title rendered in the pane's corner label."""

        return self._impl.props.title

    @title.setter
    def title(self, value: str) -> None:
        self._check_not_removed()
        if not isinstance(value, str):
            raise TypeError("Viewport pane title must be a string.")
        with self._impl.api._lock:
            self._check_not_removed()
            if value == self._impl.props.title:
                return
            self._impl.props.title = value
            self._queue_update({"title": value})

    @property
    def visible(self) -> bool:
        """Whether this pane is visible."""

        return self._impl.props.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._check_not_removed()
        if not isinstance(value, bool):
            raise TypeError("Viewport pane visibility must be a boolean.")
        with self._impl.api._lock:
            self._check_not_removed()
            if value == self._impl.props.visible:
                return
            self._impl.props.visible = value
            self._queue_update({"visible": value})

    @property
    def fit(self) -> ViewportPaneFit:
        """How the image is sized within its pane."""

        return self._impl.props.fit

    @fit.setter
    def fit(self, value: ViewportPaneFit) -> None:
        self._check_not_removed()
        value = _validate_fit(value)
        with self._impl.api._lock:
            self._check_not_removed()
            if value == self._impl.props.fit:
                return
            self._impl.props.fit = value
            self._queue_update({"fit": value})

    def remove(self) -> None:
        """Permanently remove this pane from the viewport."""

        api = self._impl.api
        with api._lock:
            if self._impl.removed:
                warnings.warn(
                    f"Attempted to remove an already removed {type(self).__name__}.",
                    stacklevel=2,
                )
                return
            self._impl.removed = True
            api._handle_from_pane_id.pop(self._impl.pane_id, None)
            api._websock_interface.queue_message(
                _messages.ViewportPaneRemoveMessage(pane_id=self._impl.pane_id)
            )
            api._queue_snapshot()


class ViewportApi:
    """Interface for native panes in the browser-managed viewport workspace."""

    scene_pane_id: Literal["scene"] = "scene"
    """Stable identifier for the built-in 3D scene pane."""

    def __init__(self, owner: ViserServer) -> None:
        self._lock = threading.RLock()
        self._websock_interface = owner._websock_server
        self._handle_from_pane_id: dict[str, ViewportImageHandle] = {}
        self._scene_visible = True
        self._queue_snapshot()

    @property
    def scene_visible(self) -> bool:
        """Whether the 3D scene pane is shown when 2D panes are available."""

        return self._scene_visible

    @scene_visible.setter
    def scene_visible(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Viewport scene visibility must be a boolean.")
        with self._lock:
            if value == self._scene_visible:
                return
            self._scene_visible = value
            self._websock_interface.queue_message(
                _messages.ViewportPaneUpdateMessage(
                    pane_id=self.scene_pane_id,
                    updates={"visible": value},
                )
            )

    def _known_pane_ids(self) -> tuple[str, ...]:
        """Return pane IDs in declaration order, excluding the scene pane."""

        with self._lock:
            return tuple(self._handle_from_pane_id)

    def _visible_pane_ids(self) -> set[str]:
        return {self.scene_pane_id} | {
            handle.pane_id
            for handle in self._handle_from_pane_id.values()
            if handle.visible
        }

    def _queue_snapshot(self) -> None:
        """Queue the authoritative pane registry after lifecycle messages."""

        self._websock_interface.queue_message(
            _messages.ViewportPaneSnapshotMessage(pane_ids=self._known_pane_ids())
        )

    def add_image(
        self,
        image: np.ndarray,
        *,
        pane_id: str | None = None,
        title: str = "Image",
        format: Literal["auto", "png", "jpeg"] = "auto",
        jpeg_quality: int | None = None,
        fit: ViewportPaneFit = "contain",
        visible: bool = True,
        placement: ViewportPanePlacement = "right",
        relative_to: str = "scene",
    ) -> ViewportImageHandle:
        """Add a native image pane to the viewport workspace.

        The browser owns pane arrangement and persists it locally. Placement
        and relative_to are only used when the browser first encounters a pane
        that is not already present in its saved layout.

        Args:
            image: RGB or RGBA image with shape (height, width, 3|4).
            pane_id: Stable identifier for browser layout persistence. By
                default a UUID is generated. Set this explicitly to restore a
                pane's position after a server restart.
            title: Pane corner-label title.
            format: Transport encoding. "auto" chooses PNG for RGBA and JPEG
                for RGB.
            jpeg_quality: JPEG encoder quality from 0 to 100.
            fit: Image sizing policy within the pane.
            visible: Initial visibility.
            placement: Initial split edge relative to relative_to.
            relative_to: Visible pane used for initial placement.

        Returns:
            Handle for updating or removing the image pane.
        """

        image = _validate_image(image)
        if pane_id is None:
            pane_id = str(uuid.uuid4())
        elif not isinstance(pane_id, str):
            raise TypeError("Viewport pane ID must be a string.")
        elif not pane_id:
            raise ValueError("Viewport pane ID must not be empty.")
        if pane_id == self.scene_pane_id:
            raise ValueError(f"Viewport pane ID {pane_id!r} is reserved.")
        if not isinstance(title, str):
            raise TypeError("Viewport pane title must be a string.")
        if format not in ("auto", "png", "jpeg"):
            raise ValueError("format must be 'auto', 'png', or 'jpeg'.")
        if jpeg_quality is not None and (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 0 <= jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer from 0 to 100.")
        fit = _validate_fit(fit)
        if not isinstance(visible, bool):
            raise TypeError("Viewport pane visibility must be a boolean.")
        if placement not in ("left", "right", "top", "bottom"):
            raise ValueError("placement must be left, right, top, or bottom.")
        if not isinstance(relative_to, str):
            raise TypeError("relative_to must be a viewport pane ID string.")
        if format == "jpeg" and image.shape[2] == 4:
            warnings.warn(
                "Encoding an RGBA viewport image as JPEG discards its alpha channel.",
                stacklevel=2,
            )

        resolved_format, data = _encode_image_binary(
            image, format, jpeg_quality=jpeg_quality
        )
        props = _messages.ViewportImageProps(
            _data=data,
            _format=resolved_format,
            title=title,
            visible=visible,
            fit=fit,
        )
        handle = ViewportImageHandle(
            _ViewportImageHandleState(
                pane_id=pane_id,
                props=copy.deepcopy(props),
                api=self,
                image=image.copy(),
                requested_format=format,
                jpeg_quality=jpeg_quality,
            )
        )

        with self._lock:
            if pane_id in self._handle_from_pane_id:
                raise ValueError(f"Viewport pane ID {pane_id!r} already exists.")
            if relative_to not in self._visible_pane_ids():
                raise ValueError(
                    f"Unknown or hidden relative viewport pane ID: {relative_to!r}."
                )
            self._handle_from_pane_id[pane_id] = handle
            self._websock_interface.queue_message(
                _messages.ViewportImageMessage(
                    pane_id=pane_id,
                    props=props,
                    placement=placement,
                    relative_to=relative_to,
                )
            )
            self._queue_snapshot()
        return handle


def _validate_fit(value: object) -> ViewportPaneFit:
    if value not in ("contain", "cover", "fill"):
        raise ValueError("fit must be 'contain', 'cover', or 'fill'.")
    return cast(ViewportPaneFit, value)


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            "Viewport images must have shape (height, width, 3) for RGB or "
            "(height, width, 4) for RGBA."
        )
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError("Viewport images must use an integer or floating dtype.")
    return image
