Viewport API
============

The viewport is a browser-arrangeable workspace containing the permanent 3D
scene pane and optional native 2D panes. Panes created through
``server.viewport`` are shared by all connected clients. The regular GUI
panel remains separate and can control either 2D or 3D content through normal
callbacks.

Image panes
-----------

Create an image pane beside the scene and update it in place for video-like
streams:

.. code-block:: python

   image = server.viewport.add_image(
       frame,
       pane_id="camera-feed",
       title="Camera",
       placement="right",
       fit="contain",
   )
   image.image = next_frame

The viewport behaves like an editor workspace. Drag a pane's corner label near
another pane's edge to split or move it, or drop in the center to swap the two
panes. Shared dividers resize adjacent panes while snapping to a global square
grid, so pane edges and corners always land on grid points. Dividers also
support arrow-key resizing, and a
focused pane label can swap with a neighbor using ``Shift`` plus an arrow key.

Hiding a pane collapses its split and expands the remaining panes. Showing it
again inserts it back into the workspace.

See the :download:`complete streaming example
<../../../../examples/02_gui/11_viewport_images.py>`.

Layout persistence
------------------

Each browser owns its arrangement. Viser saves the layout in browser-local
storage, scoped to the connected server URL, and restores it after a reload or
disconnect. When the browser reconnects, panes that no longer exist are removed
from the saved arrangement and new panes are inserted using their ``placement``
and ``relative_to`` hints.

Pass an explicit, stable ``pane_id`` when a pane should return to the same
position after restarting the Python process. Automatically generated IDs are
stable only for the lifetime of their handles. Layouts are local to one browser
and are not synchronized across browsers or devices.

Reference
---------

.. autoclass:: viser.ViewportApi
   :members:
   :undoc-members:

.. autoclass:: viser.ViewportImageHandle
   :members:
   :undoc-members:
