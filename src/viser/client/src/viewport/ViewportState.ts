import React from "react";

import { createStore, Store } from "../store";
import {
  VIEWPORT_SCENE_PANE_ID,
  ViewportDropRegion,
  ViewportLayout,
  collectViewportPaneIds,
  hasViewportPane,
  insertViewportPane,
  normalizeViewportLayout,
  reconcileViewportLayout,
  removeViewportPane,
  sameViewportLayout,
} from "./layoutModel";

export type ViewportImageFit = "contain" | "cover" | "fill";
export type ViewportPanePlacement = Exclude<ViewportDropRegion, "center">;

export interface ViewportImageProps {
  _data: Uint8Array<ArrayBuffer> | null;
  _format: "jpeg" | "png";
  title: string;
  visible: boolean;
  fit: ViewportImageFit;
}

export interface ViewportScenePane {
  kind: "scene";
  paneId: typeof VIEWPORT_SCENE_PANE_ID;
}

export interface ViewportImagePane {
  kind: "image";
  paneId: string;
  props: ViewportImageProps;
}

export type ViewportPane = ViewportScenePane | ViewportImagePane;

export interface ViewportState {
  panes: Record<string, ViewportPane>;
  layout: ViewportLayout;
  interactionEpoch: number;
}

export interface ViewportImageDeclaration {
  pane_id: string;
  props: ViewportImageProps;
  placement: ViewportPanePlacement;
  relative_to: string;
}

export interface ViewportActions {
  /** Clear all temporal pane state, including layout (used by file playback). */
  reset: () => void;
  /** Clear pane contents for a new connection while retaining browser layout. */
  resetPanes: () => void;
  /** Select and restore the layout storage namespace for a websocket server. */
  setPersistenceServer: (serverUrl: string) => void;
  addImagePane: (message: ViewportImageDeclaration) => void;
  updatePane: (paneId: string, updates: Partial<ViewportImageProps>) => void;
  removePane: (paneId: string) => void;
  setPaneSnapshot: (paneIds: readonly string[]) => void;
  commitUserLayout: (layout: ViewportLayout) => void;
}

export interface ViewportLayoutStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const STORAGE_KEY_PREFIX = "viser.viewport.layout.v1:";

export function viewportLayoutStorageKey(serverUrl: string): string {
  return STORAGE_KEY_PREFIX + serverUrl;
}

function browserStorage(): ViewportLayoutStorage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function initialLayout(): ViewportLayout {
  return {
    version: 1,
    root: { type: "pane", pane_id: VIEWPORT_SCENE_PANE_ID },
  };
}

function initialPanes(): Record<string, ViewportPane> {
  const panes = Object.create(null) as Record<string, ViewportPane>;
  panes[VIEWPORT_SCENE_PANE_ID] = {
    kind: "scene",
    paneId: VIEWPORT_SCENE_PANE_ID,
  };
  return panes;
}

function initialState(
  layout = initialLayout(),
  interactionEpoch = 0,
): ViewportState {
  return { panes: initialPanes(), layout, interactionEpoch };
}

function copyPaneRecord(
  panes: Record<string, ViewportPane>,
): Record<string, ViewportPane> {
  return Object.assign(
    Object.create(null) as Record<string, ViewportPane>,
    panes,
  );
}

function paneIsVisible(pane: ViewportPane | undefined): boolean {
  return pane === undefined || pane.kind === "scene" || pane.props.visible;
}

function retainedPaneIds(
  layout: ViewportLayout,
  panes: Record<string, ViewportPane>,
  authoritativePaneIds: ReadonlySet<string> | null,
): string[] {
  return collectViewportPaneIds(layout).filter(
    (paneId) =>
      paneId === VIEWPORT_SCENE_PANE_ID ||
      ((authoritativePaneIds === null || authoritativePaneIds.has(paneId)) &&
        paneIsVisible(panes[paneId])),
  );
}

/** Per-viewer pane registry and browser-owned layout store. */
export function useViewportState(
  storage: ViewportLayoutStorage | null = browserStorage(),
): {
  store: Store<ViewportState>;
  actions: ViewportActions;
} {
  const store = React.useMemo(() => createStore(initialState()), []);
  const storageKeyRef = React.useRef<string | null>(null);
  const authoritativePaneIdsRef = React.useRef<Set<string> | null>(null);

  const actions = React.useMemo<ViewportActions>(() => {
    const persistLayout = (layout: ViewportLayout): void => {
      const storageKey = storageKeyRef.current;
      if (storage === null || storageKey === null) return;
      try {
        storage.setItem(storageKey, JSON.stringify(layout));
      } catch {
        // Storage can be unavailable or full. Layout remains usable in memory.
      }
    };

    const commitLayout = (layout: ViewportLayout): boolean => {
      if (sameViewportLayout(layout, store.get().layout)) return false;
      persistLayout(layout);
      return true;
    };

    return {
      reset: () => {
        authoritativePaneIdsRef.current = null;
        store.set(
          initialState(initialLayout(), store.get().interactionEpoch + 1),
        );
      },

      resetPanes: () => {
        authoritativePaneIdsRef.current = null;
        store.set((state) => ({
          panes: initialPanes(),
          interactionEpoch: state.interactionEpoch + 1,
        }));
      },

      setPersistenceServer: (serverUrl) => {
        const storageKey = viewportLayoutStorageKey(serverUrl);
        if (storageKeyRef.current === storageKey) return;
        storageKeyRef.current = storageKey;
        authoritativePaneIdsRef.current = null;

        let layout = initialLayout();
        if (storage !== null) {
          try {
            const serialized = storage.getItem(storageKey);
            if (serialized !== null) {
              layout = normalizeViewportLayout(JSON.parse(serialized));
              storage.setItem(storageKey, JSON.stringify(layout));
            }
          } catch {
            // Malformed or inaccessible storage falls back to the scene pane.
          }
        }
        store.set(initialState(layout, store.get().interactionEpoch + 1));
      },

      addImagePane: (message) => {
        const paneId = message.pane_id;
        if (paneId === VIEWPORT_SCENE_PANE_ID || paneId.length === 0) return;
        authoritativePaneIdsRef.current?.add(paneId);

        const state = store.get();
        const panes = copyPaneRecord(state.panes);
        panes[paneId] = {
          kind: "image",
          paneId,
          props: message.props,
        };

        let layout = state.layout;
        if (!message.props.visible) {
          layout = removeViewportPane(layout, paneId);
        } else if (!hasViewportPane(layout, paneId)) {
          const relativeTo = hasViewportPane(layout, message.relative_to)
            ? message.relative_to
            : VIEWPORT_SCENE_PANE_ID;
          layout = insertViewportPane(
            layout,
            paneId,
            relativeTo,
            message.placement,
          );
        }

        if (commitLayout(layout)) store.set({ panes, layout });
        else store.set({ panes });
      },

      updatePane: (paneId, updates) => {
        const state = store.get();
        const pane = state.panes[paneId];
        if (pane === undefined || pane.kind !== "image") return;
        const panes = copyPaneRecord(state.panes);
        const updatedPane = { ...pane, props: { ...pane.props, ...updates } };
        panes[paneId] = updatedPane;

        let layout = state.layout;
        if (updatedPane.props.visible !== pane.props.visible) {
          layout = updatedPane.props.visible
            ? insertViewportPane(
                layout,
                paneId,
                VIEWPORT_SCENE_PANE_ID,
                "right",
              )
            : removeViewportPane(layout, paneId);
        }

        if (commitLayout(layout)) store.set({ panes, layout });
        else store.set({ panes });
      },

      removePane: (paneId) => {
        if (paneId === VIEWPORT_SCENE_PANE_ID) return;
        authoritativePaneIdsRef.current?.delete(paneId);
        const state = store.get();
        const panes = copyPaneRecord(state.panes);
        delete panes[paneId];
        const layout = removeViewportPane(state.layout, paneId);
        if (commitLayout(layout)) store.set({ panes, layout });
        else store.set({ panes });
      },

      setPaneSnapshot: (paneIds) => {
        const authoritativePaneIds = new Set(
          paneIds.filter(
            (paneId) =>
              paneId.length > 0 && paneId !== VIEWPORT_SCENE_PANE_ID,
          ),
        );
        authoritativePaneIdsRef.current = authoritativePaneIds;

        const state = store.get();
        const panes = copyPaneRecord(state.panes);
        Object.keys(panes).forEach((paneId) => {
          if (
            paneId !== VIEWPORT_SCENE_PANE_ID &&
            !authoritativePaneIds.has(paneId)
          ) {
            delete panes[paneId];
          }
        });

        // Retain saved leaves named by the snapshot even if their create has
        // not arrived yet, but do not invent leaves for unhydrated IDs.
        const layout = reconcileViewportLayout(
          state.layout,
          retainedPaneIds(state.layout, panes, authoritativePaneIds),
        );
        if (commitLayout(layout)) store.set({ panes, layout });
        else store.set({ panes });
      },

      commitUserLayout: (rawLayout) => {
        const state = store.get();
        const normalized = normalizeViewportLayout(rawLayout);
        const layout = reconcileViewportLayout(
          normalized,
          retainedPaneIds(
            normalized,
            state.panes,
            authoritativePaneIdsRef.current,
          ),
        );
        if (commitLayout(layout)) store.set({ layout });
      },
    };
  }, [storage, store]);

  return { store, actions };
}
