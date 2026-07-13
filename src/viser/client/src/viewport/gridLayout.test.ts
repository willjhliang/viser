import { describe, expect, it } from "vitest";

import { ViewportLayout } from "./layoutModel";
import {
  computeLayoutGeometry,
  directionalPaneTarget,
  dropRegionForPoint,
  gridSpecForLayout,
  resizeLayoutAtGridLine,
} from "./gridLayout";

const nestedLayout: ViewportLayout = {
  version: 1,
  root: {
    type: "split",
    direction: "row",
    children: [
      { type: "pane", pane_id: "scene" },
      {
        type: "split",
        direction: "column",
        children: [
          { type: "pane", pane_id: "video" },
          { type: "pane", pane_id: "plot" },
        ],
        weights: [0.5, 0.5],
      },
    ],
    weights: [0.5, 0.5],
  },
};

describe("viewport square-grid geometry", () => {
  it("represents common aspect ratios with integer grid corners", () => {
    const grid = gridSpecForLayout(nestedLayout, {
      width: 1024,
      height: 576,
    });
    expect(grid).toEqual({ columns: 64, rows: 36, cellSize: 16 });

    const geometry = computeLayoutGeometry(
      nestedLayout.root,
      grid.columns,
      grid.rows,
    );
    expect(geometry.panes.video).toEqual({
      x: 32,
      y: 0,
      width: 32,
      height: 18,
    });
    expect(
      geometry.panes.video.width / geometry.panes.video.height,
    ).toBeCloseTo(16 / 9);
    expect(
      geometry.dividers.every((divider) =>
        Number.isInteger(divider.coordinate),
      ),
    ).toBe(true);
  });

  it("snaps divider updates to grid lines and clamps subtree minima", () => {
    const grid = gridSpecForLayout(nestedLayout, {
      width: 1024,
      height: 576,
    });
    const geometry = computeLayoutGeometry(
      nestedLayout.root,
      grid.columns,
      grid.rows,
    );
    const rootDivider = geometry.dividers.find(
      (divider) => divider.direction === "row",
    )!;

    const moved = resizeLayoutAtGridLine(nestedLayout, rootDivider, 20);
    expect(moved.gridLine).toBe(20);
    expect(moved.layout.root).toMatchObject({ weights: [20 / 64, 44 / 64] });

    const clamped = resizeLayoutAtGridLine(nestedLayout, rootDivider, -100);
    expect(clamped.gridLine).toBe(4);
  });

  it("classifies edge and center pane drops", () => {
    const rect = { x: 10, y: 20, width: 40, height: 20 };
    expect(dropRegionForPoint(rect, 11, 30)).toBe("left");
    expect(dropRegionForPoint(rect, 49, 30)).toBe("right");
    expect(dropRegionForPoint(rect, 30, 21)).toBe("top");
    expect(dropRegionForPoint(rect, 30, 39)).toBe("bottom");
    expect(dropRegionForPoint(rect, 30, 30)).toBe("center");
  });

  it("chooses deterministic directional keyboard targets", () => {
    const panes = {
      scene: { x: 0, y: 0, width: 20, height: 20 },
      upper: { x: 20, y: 0, width: 20, height: 10 },
      lower: { x: 20, y: 10, width: 20, height: 10 },
    };
    expect(directionalPaneTarget(panes, "scene", "right")).toBe("lower");
    expect(directionalPaneTarget(panes, "upper", "bottom")).toBe("lower");
    expect(directionalPaneTarget(panes, "scene", "left")).toBeNull();
  });
});
