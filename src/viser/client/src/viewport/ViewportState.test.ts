import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  ViewportLayout,
  collectViewportPaneIds,
  dropViewportPane,
} from "./layoutModel";
import {
  ViewportActions,
  ViewportImageDeclaration,
  ViewportLayoutStorage,
  ViewportState,
  useViewportState,
  viewportLayoutStorageKey,
} from "./ViewportState";

class MemoryStorage implements ViewportLayoutStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

function createViewportHarness(storage: ViewportLayoutStorage | null = null): {
  actions: ViewportActions;
  getState: () => ViewportState;
} {
  let viewport: ReturnType<typeof useViewportState> | undefined;

  function Harness(): React.ReactNode {
    viewport = useViewportState(storage);
    return null;
  }

  renderToStaticMarkup(React.createElement(Harness));
  if (viewport === undefined) {
    throw new Error("Viewport harness did not render");
  }
  const { actions, store } = viewport;
  return { actions, getState: store.get };
}

function imageDeclaration(
  paneId: string,
  overrides: Partial<ViewportImageDeclaration> = {},
): ViewportImageDeclaration {
  return {
    pane_id: paneId,
    props: {
      _data: null,
      _format: "png",
      title: paneId,
      visible: true,
      fit: "contain",
    },
    placement: "right",
    relative_to: "scene",
    ...overrides,
  };
}

const sceneAndImageLayout: ViewportLayout = {
  version: 1,
  root: {
    type: "split",
    direction: "row",
    children: [
      { type: "pane", pane_id: "scene" },
      { type: "pane", pane_id: "image" },
    ],
    weights: [0.3, 0.7],
  },
};

describe("useViewportState browser persistence", () => {
  it("restores layouts per server URL and keeps them across pane resets", () => {
    const storage = new MemoryStorage();
    const { actions, getState } = createViewportHarness(storage);

    actions.setPersistenceServer("ws://server-a");
    actions.addImagePane(imageDeclaration("image"));
    const swapped = dropViewportPane(
      getState().layout,
      "scene",
      "image",
      "center",
    );
    actions.commitUserLayout(swapped);
    const interactionEpoch = getState().interactionEpoch;
    actions.resetPanes();
    expect(getState().interactionEpoch).toBe(interactionEpoch + 1);
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "image",
      "scene",
    ]);

    actions.setPersistenceServer("ws://server-b");
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);

    actions.setPersistenceServer("ws://server-a");
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "image",
      "scene",
    ]);
    expect(getState().panes.image).toBeUndefined();
  });

  it("repairs malformed stored layouts without throwing", () => {
    const storage = new MemoryStorage();
    storage.values.set(viewportLayoutStorageKey("ws://broken"), "not json");
    const { actions, getState } = createViewportHarness(storage);

    actions.setPersistenceServer("ws://broken");

    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
  });
});

describe("useViewportState snapshot reconciliation", () => {
  it("converges when create arrives before snapshot", () => {
    const { actions, getState } = createViewportHarness();

    actions.resetPanes();
    actions.addImagePane(imageDeclaration("image"));
    actions.setPaneSnapshot(["image"]);

    expect(getState().panes.image?.kind).toBe("image");
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "scene",
      "image",
    ]);
  });

  it("retains saved placement when snapshot arrives before create", () => {
    const storage = new MemoryStorage();
    storage.values.set(
      viewportLayoutStorageKey("ws://server"),
      JSON.stringify(sceneAndImageLayout),
    );
    const { actions, getState } = createViewportHarness(storage);
    actions.setPersistenceServer("ws://server");
    actions.resetPanes();

    actions.setPaneSnapshot(["image"]);
    expect(getState().panes.image).toBeUndefined();
    expect(getState().layout).toEqual(sceneAndImageLayout);

    actions.addImagePane(imageDeclaration("image"));
    expect(getState().layout).toEqual(sceneAndImageLayout);
    expect(getState().panes.image?.kind).toBe("image");
  });

  it("prunes stale leaves without inventing unhydrated snapshot panes", () => {
    const storage = new MemoryStorage();
    storage.values.set(
      viewportLayoutStorageKey("ws://server"),
      JSON.stringify(sceneAndImageLayout),
    );
    const { actions, getState } = createViewportHarness(storage);
    actions.setPersistenceServer("ws://server");
    actions.resetPanes();

    actions.setPaneSnapshot(["future"]);
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);

    actions.addImagePane(
      imageDeclaration("future", {
        placement: "bottom",
        relative_to: "scene",
      }),
    );
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "scene",
      "future",
    ]);
    expect(getState().layout.root).toMatchObject({
      type: "split",
      direction: "column",
    });
  });

  it("treats each snapshot as exact authority", () => {
    const { actions, getState } = createViewportHarness();
    actions.resetPanes();

    actions.addImagePane(imageDeclaration("removed"));
    actions.setPaneSnapshot([]);

    expect(getState().panes.removed).toBeUndefined();
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
  });
});

describe("useViewportState pane lifecycle", () => {
  it("updates content and removes panes without layout protocol state", () => {
    const { actions, getState } = createViewportHarness();
    actions.addImagePane(imageDeclaration("image"));

    actions.updatePane("image", { title: "updated", fit: "cover" });
    const pane = getState().panes.image;
    expect(pane?.kind === "image" ? pane.props.title : undefined).toBe(
      "updated",
    );
    expect(pane?.kind === "image" ? pane.props.fit : undefined).toBe("cover");

    actions.removePane("image");
    expect(getState().panes.image).toBeUndefined();
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
  });

  it("removes hidden panes and deterministically restores visible panes", () => {
    const { actions, getState } = createViewportHarness();
    actions.addImagePane(imageDeclaration("image"));

    actions.updatePane("image", { visible: false });
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);

    actions.updatePane("image", { visible: true });
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "scene",
      "image",
    ]);
  });

  it("hides the scene before images arrive and uses it as an empty fallback", () => {
    const { actions, getState } = createViewportHarness();

    actions.updatePane("scene", { visible: false });
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
    const scenePane = getState().panes.scene;
    expect(scenePane?.kind === "scene" ? scenePane.visible : undefined).toBe(
      false,
    );

    actions.addImagePane(imageDeclaration("first"));
    expect(collectViewportPaneIds(getState().layout)).toEqual(["first"]);

    actions.updatePane("first", { visible: false });
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
    actions.updatePane("first", { visible: true });
    expect(collectViewportPaneIds(getState().layout)).toEqual(["first"]);

    actions.addImagePane(
      imageDeclaration("second", { relative_to: "scene" }),
    );
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "first",
      "second",
    ]);

    actions.removePane("first");
    actions.removePane("second");
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);

    actions.addImagePane(imageDeclaration("replacement"));
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "replacement",
    ]);
  });

  it("hides and restores the scene after images arrive", () => {
    const { actions, getState } = createViewportHarness();
    actions.addImagePane(imageDeclaration("image"));

    actions.updatePane("scene", { visible: false });
    expect(collectViewportPaneIds(getState().layout)).toEqual(["image"]);

    actions.updatePane("scene", { visible: true });
    expect(collectViewportPaneIds(getState().layout)).toEqual([
      "scene",
      "image",
    ]);
  });

  it("ignores invalid and reserved pane IDs", () => {
    const { actions, getState } = createViewportHarness();

    actions.addImagePane(imageDeclaration(""));
    actions.addImagePane(imageDeclaration("scene"));
    actions.removePane("scene");

    expect(Object.keys(getState().panes)).toEqual(["scene"]);
    expect(collectViewportPaneIds(getState().layout)).toEqual(["scene"]);
  });
});
