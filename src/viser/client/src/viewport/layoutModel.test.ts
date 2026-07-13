import { describe, expect, it } from "vitest";

import {
  VIEWPORT_SCENE_PANE_ID,
  ViewportLayout,
  ViewportLayoutNode,
  ViewportSplitDirection,
  collectViewportPaneIds,
  dropViewportPane,
  insertViewportPane,
  normalizeViewportWeights,
  reconcileViewportLayout,
  removeViewportPane,
  setViewportSplitWeights,
} from "./layoutModel";

const pane = (pane_id: string): ViewportLayoutNode => ({
  type: "pane",
  pane_id,
});
const split = (
  direction: ViewportSplitDirection,
  children: ViewportLayoutNode[],
  weights = children.map(() => 1 / children.length),
): ViewportLayoutNode => ({ type: "split", direction, children, weights });
const layout = (root: ViewportLayoutNode): ViewportLayout => ({
  version: 1,
  root,
});

function expectValidLayout(value: ViewportLayout): void {
  const seen = new Set<string>();
  const visit = (node: ViewportLayoutNode): void => {
    if (node.type === "pane") {
      expect(seen.has(node.pane_id)).toBe(false);
      seen.add(node.pane_id);
      return;
    }
    expect(node.children.length).toBeGreaterThanOrEqual(2);
    expect(node.weights).toHaveLength(node.children.length);
    expect(
      node.weights.every((weight) => Number.isFinite(weight) && weight > 0),
    ).toBe(true);
    expect(node.weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(
      1,
    );
    node.children.forEach(visit);
  };
  expect(value.version).toBe(1);
  visit(value.root);
  expect(seen.has(VIEWPORT_SCENE_PANE_ID)).toBe(true);
}

describe("normalizeViewportWeights", () => {
  it("repairs bad entries and handles huge finite weights", () => {
    expect(normalizeViewportWeights([2, -1, 2])).toEqual([0.4, 0.2, 0.4]);
    expect(normalizeViewportWeights([], 2)).toEqual([0.5, 0.5]);
    expect(normalizeViewportWeights([1e308, 1e308])).toEqual([0.5, 0.5]);
  });

  it("keeps extreme positive weights positive and renormalizable", () => {
    const normalized = normalizeViewportWeights([1e308, Number.MIN_VALUE]);

    expect(normalized).toHaveLength(2);
    expect(normalized.every((weight) => Number.isFinite(weight))).toBe(true);
    expect(normalized.every((weight) => weight > 0)).toBe(true);
    expect(normalized.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1);
    expect(normalizeViewportWeights(normalized)).toEqual(normalized);
  });

  it("rejects invalid child counts", () => {
    expect(normalizeViewportWeights([1], -1)).toEqual([]);
    expect(normalizeViewportWeights([1], 1.5)).toEqual([]);
  });
});

describe("reconcileViewportLayout", () => {
  it("prunes unknown/duplicate panes, repairs weights, and appends missing panes", () => {
    const input = {
      version: 1,
      root: {
        type: "split",
        direction: "row",
        children: [
          { type: "pane", pane_id: "unknown" },
          { type: "pane", pane_id: "image" },
          { type: "pane", pane_id: "image" },
          {
            type: "split",
            direction: "column",
            children: [{ type: "invalid" }, { type: "pane", pane_id: "scene" }],
            weights: [-1, 2],
          },
        ],
        weights: [9, 2, 3, 4],
      },
    };
    const before = JSON.stringify(input);
    const output = reconcileViewportLayout(input, ["image", "plot"]);

    expect(collectViewportPaneIds(output)).toEqual(["image", "scene", "plot"]);
    expect(output.root).toMatchObject({
      type: "split",
      direction: "row",
      weights: [1 / 3, 1 / 3, 1 / 3],
    });
    expect(JSON.stringify(input)).toBe(before);
    expectValidLayout(output);
  });

  it("retains scene and produces deterministic defaults", () => {
    expect(
      collectViewportPaneIds(
        reconcileViewportLayout(
          { version: 1, root: { type: "pane", pane_id: "image" } },
          ["image"],
        ),
      ),
    ).toEqual(["scene", "image"]);
    expect(
      collectViewportPaneIds(reconcileViewportLayout(null, ["b", "a", "b"])),
    ).toEqual(["scene", "b", "a"]);
    expect(
      collectViewportPaneIds(
        reconcileViewportLayout(
          { version: 2, root: { type: "pane", pane_id: "a" } },
          ["a"],
        ),
      ),
    ).toEqual(["scene", "a"]);
  });

  it("omits a hidden scene but retains it as the empty fallback", () => {
    const sceneAndImage = layout(
      split("row", [pane("scene"), pane("image")]),
    );
    expect(
      collectViewportPaneIds(
        reconcileViewportLayout(
          sceneAndImage,
          ["image"],
          VIEWPORT_SCENE_PANE_ID,
          false,
        ),
      ),
    ).toEqual(["image"]);
    expect(
      collectViewportPaneIds(
        reconcileViewportLayout(
          layout(pane("scene")),
          [],
          VIEWPORT_SCENE_PANE_ID,
          false,
        ),
      ),
    ).toEqual(["scene"]);
  });

  it("preserves nested same-axis splits from serialized layouts", () => {
    const output = reconcileViewportLayout(
      {
        version: 1,
        root: {
          type: "split",
          direction: "row",
          children: [
            { type: "pane", pane_id: "scene" },
            {
              type: "split",
              direction: "row",
              children: [
                { type: "pane", pane_id: "a" },
                { type: "pane", pane_id: "b" },
              ],
              weights: [3, 1],
            },
          ],
          weights: [1, 1],
        },
      },
      ["a", "b"],
    );
    expect(output.root).toEqual(
      split(
        "row",
        [pane("scene"), split("row", [pane("a"), pane("b")], [0.75, 0.25])],
        [0.5, 0.5],
      ),
    );
    expectValidLayout(output);
  });

  it("safely discards cyclic object graphs", () => {
    const cyclic: {
      type: "split";
      direction: "column";
      children: unknown[];
      weights: number[];
    } = { type: "split", direction: "column", children: [], weights: [1, 1] };
    cyclic.children.push(cyclic, { type: "pane", pane_id: "scene" });
    const output = reconcileViewportLayout({ version: 1, root: cyclic }, [
      "image",
    ]);
    expect(collectViewportPaneIds(output)).toEqual(["scene", "image"]);
    expectValidLayout(output);
  });
});

describe("removeViewportPane", () => {
  const nested = (): ViewportLayout =>
    layout(
      split(
        "row",
        [
          pane("scene"),
          split("column", [pane("a"), pane("b")], [0.7, 0.3]),
          pane("c"),
        ],
        [0.2, 0.5, 0.3],
      ),
    );

  it("removes panes and collapses one-child splits", () => {
    const output = removeViewportPane(nested(), "b");
    expect(collectViewportPaneIds(output)).toEqual(["scene", "a", "c"]);
    expect(output.root).toMatchObject({ weights: [0.2, 0.5, 0.3] });
    expectValidLayout(output);
  });

  it("protects scene and no-ops for unknown panes", () => {
    const input = nested();
    expect(removeViewportPane(input, "missing")).toBe(input);
    expect(removeViewportPane(input, VIEWPORT_SCENE_PANE_ID)).toBe(input);
  });

  it("collapses the root after removing the final image", () => {
    const input = layout(split("row", [pane("scene"), pane("image")]));
    expect(removeViewportPane(input, "image")).toEqual(layout(pane("scene")));
  });
});

describe("insertViewportPane and edge drops", () => {
  it("creates row and column splits around targets", () => {
    const withImage = insertViewportPane(
      layout(pane("scene")),
      "image",
      "scene",
      "right",
    );
    const output = insertViewportPane(withImage, "plot", "image", "bottom");
    expect(output.root).toEqual(
      split(
        "row",
        [pane("scene"), split("column", [pane("image"), pane("plot")])],
        [0.5, 0.5],
      ),
    );
    expectValidLayout(output);
  });

  it("splits a target allocation when adding a same-axis sibling", () => {
    const input = layout(
      split("row", [pane("scene"), pane("image")], [0.8, 0.2]),
    );
    const output = insertViewportPane(input, "plot", "image", "right");
    expect(output.root).toMatchObject({ weights: [0.8, 0.1, 0.1] });
    expect(collectViewportPaneIds(output)).toEqual(["scene", "image", "plot"]);
  });

  it("reorders siblings while preserving pane allocations", () => {
    const input = layout(
      split("row", [pane("scene"), pane("a"), pane("b")], [0.6, 0.3, 0.1]),
    );
    const output = insertViewportPane(input, "b", "a", "left");
    expect(collectViewportPaneIds(output)).toEqual(["scene", "b", "a"]);
    expect(output.root).toMatchObject({ weights: [0.6, 0.1, 0.3] });
  });

  it("detaches, collapses, and reinserts panes moved across axes", () => {
    const input = layout(
      split(
        "row",
        [pane("scene"), split("column", [pane("a"), pane("b")], [0.7, 0.3])],
        [0.6, 0.4],
      ),
    );
    const output = insertViewportPane(input, "a", "scene", "right");
    expect(collectViewportPaneIds(output)).toEqual(["scene", "a", "b"]);
    expect(output.root.type).toBe("split");
    if (output.root.type === "split") {
      output.root.weights.forEach((weight, index) =>
        expect(weight).toBeCloseTo([0.3, 0.3, 0.4][index]),
      );
    }
    expectValidLayout(output);
  });

  it("safely no-ops for missing and self targets", () => {
    const input = layout(split("row", [pane("scene"), pane("a")]));
    expect(insertViewportPane(input, "a", "missing", "right")).toBe(input);
    expect(insertViewportPane(input, "a", "a", "right")).toBe(input);
  });
});

describe("dropViewportPane", () => {
  it("center swaps positions without changing geometry", () => {
    const input = layout(
      split(
        "row",
        [pane("scene"), split("column", [pane("a"), pane("b")], [0.7, 0.3])],
        [0.6, 0.4],
      ),
    );
    const output = dropViewportPane(input, "scene", "b", "center");
    expect(collectViewportPaneIds(output)).toEqual(["b", "a", "scene"]);
    expect(output.root).toMatchObject({
      weights: [0.6, 0.4],
      children: [{ pane_id: "b" }, { weights: [0.7, 0.3] }],
    });
    expectValidLayout(output);
  });

  it("no-ops for self and missing center drops and delegates edge drops", () => {
    const input = layout(split("row", [pane("scene"), pane("a")]));
    expect(dropViewportPane(input, "a", "a", "center")).toBe(input);
    expect(dropViewportPane(input, "missing", "a", "center")).toBe(input);
    expect(
      collectViewportPaneIds(dropViewportPane(input, "a", "scene", "left")),
    ).toEqual(["a", "scene"]);
  });
});

describe("split weights", () => {
  it("updates nested split weights by child-index path", () => {
    const input = layout(
      split(
        "row",
        [pane("scene"), split("column", [pane("a"), pane("b")])],
        [0.6, 0.4],
      ),
    );

    const set = setViewportSplitWeights(input, [1], [3, 1]);
    expect(set.root).toMatchObject({
      children: [{}, { weights: [0.75, 0.25] }],
    });
    expect(setViewportSplitWeights(input, [9], [3, 1])).toBe(input);
    expect(setViewportSplitWeights(input, [1], [1])).toBe(input);
  });
});
