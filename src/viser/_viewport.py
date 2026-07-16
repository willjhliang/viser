"""Python API for native viewport panes."""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
import uuid
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Generic, Literal, TypeVar, cast

import numpy as np
from typing_extensions import TypeAlias

from . import _messages
from ._gui_handles import _plotly_json_with_config
from ._scene_api import _encode_image_binary

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ._viser import ViserServer


ViewportPaneFit: TypeAlias = Literal["contain", "cover", "fill"]
ViewportPanePlacement: TypeAlias = Literal["left", "right", "top", "bottom"]


def _plotly_json_for_pane(figure: go.Figure, config: Mapping[str, Any] | None) -> str:
    """Serialize a figure for a viewport pane.

    Figures bake the global default template in at construction time, so a
    figure whose template matches plotly's stock "plotly" template is treated
    as "no template chosen": the template is stripped so the client can apply
    a default matched to its current light/dark theme. Any explicitly
    assigned template, or a customized ``plotly.io.templates.default``, is
    preserved.
    """
    import plotly.io as pio

    json_str = _plotly_json_with_config(figure, config)
    template = figure.layout.template
    stock_template = pio.templates["plotly"]
    if template is None or template.to_plotly_json() == stock_template.to_plotly_json():
        plot_dict = json.loads(json_str)
        plot_dict.get("layout", {}).pop("template", None)
        json_str = json.dumps(plot_dict)
    return json_str


def _plotly_theme_templates_json() -> str:
    """Themed default templates for figures that don't specify one. The
    client picks by its current color scheme: "plotly_white" when viser is in
    light mode, "plotly_dark" in dark mode."""
    import plotly.io as pio

    return json.dumps(
        {
            "light": pio.templates["plotly_white"].to_plotly_json(),
            "dark": pio.templates["plotly_dark"].to_plotly_json(),
        }
    )


@dataclasses.dataclass
class _ViewportImageHandleState:
    pane_id: str
    props: _messages.ViewportImageProps
    api: ViewportApi
    image: np.ndarray
    requested_format: Literal["auto", "jpeg", "png"]
    jpeg_quality: int | None
    removed: bool = False


@dataclasses.dataclass
class _ViewportPlotlyHandleState:
    pane_id: str
    props: _messages.ViewportPlotlyProps
    api: ViewportApi
    figure: go.Figure
    config: Mapping[str, Any] | None
    removed: bool = False


_PaneStateT = TypeVar(
    "_PaneStateT", _ViewportImageHandleState, _ViewportPlotlyHandleState
)


class _ViewportPaneHandle(Generic[_PaneStateT]):
    """Lifecycle and property logic shared by all viewport pane handles."""

    _impl: _PaneStateT

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


class ViewportImageHandle(_ViewportPaneHandle[_ViewportImageHandleState]):
    """Handle for updating or removing a native viewport image pane."""

    def __init__(self, state: _ViewportImageHandleState) -> None:
        self._impl = state

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


class ViewportPlotlyHandle(_ViewportPaneHandle[_ViewportPlotlyHandleState]):
    """Handle for updating or removing a native viewport Plotly pane."""

    def __init__(self, state: _ViewportPlotlyHandleState) -> None:
        self._impl = state

    @property
    def figure(self) -> go.Figure:
        """Current Plotly figure. Assign a new figure to update the pane."""

        return self._impl.figure

    @figure.setter
    def figure(self, figure: go.Figure) -> None:
        self._check_not_removed()
        json_str = _plotly_json_for_pane(figure, self._impl.config)
        with self._impl.api._lock:
            # Serialization can be expensive. Recheck after taking the lock so
            # a concurrent remove cannot queue an update for a reused pane ID.
            self._check_not_removed()
            self._impl.figure = figure
            self._impl.props._plotly_json_str = json_str
            self._queue_update({"_plotly_json_str": json_str})


class ViewportPaneGroup:
    """Adds panes to an equally divided row or column.

    Returned by :meth:`ViewportApi.add_row` and :meth:`ViewportApi.add_column`.
    Each pane added through the group is placed along the group's axis and the
    group re-divides its combined space equally, without disturbing panes
    outside the group. The group only shapes creation-time placement: panes it
    creates are ordinary panes afterwards, and browser-saved arrangements
    still take precedence on reload.
    """

    def __init__(
        self,
        api: ViewportApi,
        axis: Literal["row", "column"],
        placement: ViewportPanePlacement,
        relative_to: str,
    ) -> None:
        self._api = api
        self._axis: Literal["row", "column"] = axis
        self._placement: ViewportPanePlacement = placement
        self._relative_to = relative_to
        self._member_ids: list[str] = []

    def _next_declaration(
        self,
    ) -> tuple[ViewportPanePlacement, str, tuple[str, ...]]:
        """Placement hints for the group's next pane."""

        if not self._member_ids:
            return self._placement, self._relative_to, ()
        placement: ViewportPanePlacement = "right" if self._axis == "row" else "bottom"
        return placement, self._member_ids[-1], tuple(self._member_ids)

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
    ) -> ViewportImageHandle:
        """Add an image pane to the group. Accepts the same arguments as
        :meth:`ViewportApi.add_image`, minus placement, which the group
        owns."""

        placement, relative_to, equalize_group = self._next_declaration()
        handle = self._api._add_image(
            image,
            pane_id=pane_id,
            title=title,
            format=format,
            jpeg_quality=jpeg_quality,
            fit=fit,
            visible=visible,
            placement=placement,
            relative_to=relative_to,
            equalize_group=equalize_group,
        )
        self._member_ids.append(handle.pane_id)
        return handle

    def add_plotly(
        self,
        figure: go.Figure,
        *,
        config: Mapping[str, Any] | None = None,
        pane_id: str | None = None,
        title: str = "Plot",
        visible: bool = True,
    ) -> ViewportPlotlyHandle:
        """Add a Plotly pane to the group. Accepts the same arguments as
        :meth:`ViewportApi.add_plotly`, minus placement, which the group
        owns."""

        placement, relative_to, equalize_group = self._next_declaration()
        handle = self._api._add_plotly(
            figure,
            config=config,
            pane_id=pane_id,
            title=title,
            visible=visible,
            placement=placement,
            relative_to=relative_to,
            equalize_group=equalize_group,
        )
        self._member_ids.append(handle.pane_id)
        return handle


class ViewportApi:
    """Interface for native panes in the browser-managed viewport workspace."""

    scene_pane_id: Literal["scene"] = "scene"
    """Stable identifier for the built-in 3D scene pane."""

    def __init__(self, owner: ViserServer) -> None:
        self._lock = threading.RLock()
        self._owner = owner
        self._websock_interface = owner._websock_server
        self._handle_from_pane_id: dict[str, _ViewportPaneHandle[Any]] = {}
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

    def _validate_pane_declaration(
        self,
        pane_id: str | None,
        title: str,
        visible: bool,
        placement: ViewportPanePlacement,
        relative_to: str,
    ) -> str:
        """Validate shared pane arguments and return the resolved pane ID."""

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
        if not isinstance(visible, bool):
            raise TypeError("Viewport pane visibility must be a boolean.")
        if placement not in ("left", "right", "top", "bottom"):
            raise ValueError("placement must be left, right, top, or bottom.")
        if not isinstance(relative_to, str):
            raise TypeError("relative_to must be a viewport pane ID string.")
        return pane_id

    def _register_pane(
        self,
        pane_id: str,
        handle: _ViewportPaneHandle[Any],
        create_message: _messages.Message,
        relative_to: str,
    ) -> None:
        """Register a new pane handle and queue its creation messages."""

        with self._lock:
            if pane_id in self._handle_from_pane_id:
                raise ValueError(f"Viewport pane ID {pane_id!r} already exists.")
            if relative_to not in self._visible_pane_ids():
                raise ValueError(
                    f"Unknown or hidden relative viewport pane ID: {relative_to!r}."
                )
            self._handle_from_pane_id[pane_id] = handle
            self._websock_interface.queue_message(create_message)
            self._queue_snapshot()

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

        return self._add_image(
            image,
            pane_id=pane_id,
            title=title,
            format=format,
            jpeg_quality=jpeg_quality,
            fit=fit,
            visible=visible,
            placement=placement,
            relative_to=relative_to,
            equalize_group=(),
        )

    def _add_image(
        self,
        image: np.ndarray,
        *,
        pane_id: str | None,
        title: str,
        format: Literal["auto", "png", "jpeg"],
        jpeg_quality: int | None,
        fit: ViewportPaneFit,
        visible: bool,
        placement: ViewportPanePlacement,
        relative_to: str,
        equalize_group: tuple[str, ...],
    ) -> ViewportImageHandle:
        image = _validate_image(image)
        pane_id = self._validate_pane_declaration(
            pane_id, title, visible, placement, relative_to
        )
        if format not in ("auto", "png", "jpeg"):
            raise ValueError("format must be 'auto', 'png', or 'jpeg'.")
        if jpeg_quality is not None and (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 0 <= jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer from 0 to 100.")
        fit = _validate_fit(fit)
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
        self._register_pane(
            pane_id,
            handle,
            _messages.ViewportImageMessage(
                pane_id=pane_id,
                props=props,
                placement=placement,
                relative_to=relative_to,
                equalize_group=equalize_group,
            ),
            relative_to,
        )
        return handle

    def add_plotly(
        self,
        figure: go.Figure,
        *,
        config: Mapping[str, Any] | None = None,
        pane_id: str | None = None,
        title: str = "Plot",
        visible: bool = True,
        placement: ViewportPanePlacement = "right",
        relative_to: str = "scene",
    ) -> ViewportPlotlyHandle:
        """Add a native interactive Plotly pane to the viewport workspace.
        Requires the `plotly` package to be installed.

        The plot is dynamically sized: it always fills its pane, including
        when panes are resized in the browser. The browser owns pane
        arrangement and persists it locally. Placement and relative_to are
        only used when the browser first encounters a pane that is not
        already present in its saved layout.

        Figures that carry plotly's stock default template are rendered with
        a template matched to each viewer's theme: "plotly_white" when viser
        is in light mode and "plotly_dark" in dark mode, tracking the
        browser's current setting live (including automatically chosen
        themes). Set any template explicitly on the figure (or change
        ``plotly.io.templates.default``) to override this.

        Args:
            figure: Plotly figure to display. Assign to the returned handle's
                ``figure`` property to update it.
            config: Plotly config dict merged into the figure JSON. Controls
                display options like ``{"displayModeBar": False}``. Values
                must be JSON-serializable. See
                https://plotly.com/javascript/configuration-options/
            pane_id: Stable identifier for browser layout persistence. By
                default a UUID is generated. Set this explicitly to restore a
                pane's position after a server restart.
            title: Pane corner-label title.
            visible: Initial visibility.
            placement: Initial split edge relative to relative_to.
            relative_to: Visible pane used for initial placement.

        Returns:
            Handle for updating or removing the Plotly pane.
        """

        return self._add_plotly(
            figure,
            config=config,
            pane_id=pane_id,
            title=title,
            visible=visible,
            placement=placement,
            relative_to=relative_to,
            equalize_group=(),
        )

    def _add_plotly(
        self,
        figure: go.Figure,
        *,
        config: Mapping[str, Any] | None,
        pane_id: str | None,
        title: str,
        visible: bool,
        placement: ViewportPanePlacement,
        relative_to: str,
        equalize_group: tuple[str, ...],
    ) -> ViewportPlotlyHandle:
        pane_id = self._validate_pane_declaration(
            pane_id, title, visible, placement, relative_to
        )

        # Clients cannot render the figure until plotly.min.js has been sent.
        # This must be queued before the pane creation message.
        self._owner.gui._ensure_plotly_js_sent()

        props = _messages.ViewportPlotlyProps(
            _plotly_json_str=_plotly_json_for_pane(figure, config),
            _theme_templates=_plotly_theme_templates_json(),
            title=title,
            visible=visible,
        )
        handle = ViewportPlotlyHandle(
            _ViewportPlotlyHandleState(
                pane_id=pane_id,
                props=copy.deepcopy(props),
                api=self,
                figure=figure,
                config=config,
            )
        )
        self._register_pane(
            pane_id,
            handle,
            _messages.ViewportPlotlyMessage(
                pane_id=pane_id,
                props=props,
                placement=placement,
                relative_to=relative_to,
                equalize_group=equalize_group,
            ),
            relative_to,
        )
        return handle

    def add_row(
        self,
        *,
        placement: ViewportPanePlacement = "right",
        relative_to: str = "scene",
    ) -> ViewportPaneGroup:
        """Create a group that lays out added panes side by side with equal
        widths.

        Panes added through the returned group's ``add_image`` and
        ``add_plotly`` methods are placed along a shared row and re-divided
        equally on each addition, so three panes yield exact thirds. The
        placement arguments position the group's first pane; like all
        placement hints, they only apply when the browser has no saved
        arrangement for these panes.

        Args:
            placement: Initial split edge for the group, relative to
                relative_to.
            relative_to: Visible pane used for the group's initial placement.

        Returns:
            Group for adding equally sized panes.
        """

        return ViewportPaneGroup(self, "row", placement, relative_to)

    def add_column(
        self,
        *,
        placement: ViewportPanePlacement = "right",
        relative_to: str = "scene",
    ) -> ViewportPaneGroup:
        """Create a group that lays out added panes stacked with equal
        heights.

        The column counterpart of :meth:`add_row`; see it for placement
        semantics.

        Args:
            placement: Initial split edge for the group, relative to
                relative_to.
            relative_to: Visible pane used for the group's initial placement.

        Returns:
            Group for adding equally sized panes.
        """

        return ViewportPaneGroup(self, "column", placement, relative_to)


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
