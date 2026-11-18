# -*- coding: utf-8 -*-
"""Conservative hierarchical occlusion culling (standard library only).

Conventions
-----------
Depth images are normalized (``data/depth.json`` uses values in ``[0, 1]``);
a *larger* value means *farther* from the camera.

Occludee boxes are mappings with normalized screen-space ``x``, ``y``, ``w``,
``h`` in ``[0, 1]`` and ``z`` = the box's nearest (front) depth.  A box is
culled only when its nearest point is **strictly farther** than *every* depth
sample under its footprint.  Equality is treated as visible, which is the
conservative choice.

Hierarchy
---------
Level 0 is the input depth image.  Level ``l + 1`` stores, for every 2x2 block
of level ``l``, the **maximum** (= farthest) depth in that block.  Odd sizes
are rounded up; cells that fall outside the real image (padding) hold
``-inf``, the neutral element of ``max``.  Padding therefore never lowers a
maximum and can never make the test less conservative.

Because every node's value is the farthest sample beneath it, a node whose
value is smaller than the box's front depth cannot contain a visible sample:
the whole subtree can be skipped.  This is what keeps queries logarithmic
instead of per-pixel.
"""

from __future__ import annotations

import math

__all__ = [
    "DepthPyramid",
    "build_pyramid",
    "cull_batch",
    "occludee_visible",
    "NEAR_PLANE",
    "VISIBLE_IN_FRONT",
    "VISIBLE_NEAR_PLANE",
    "VISIBLE_DEGENERATE",
    "VISIBLE_NO_COVERAGE",
    "CULLED_OCCLUDED",
    "CULLED_OUTSIDE_FRUSTUM",
    "UNDETERMINED_BUDGET",
]

# A single normalized near plane.  Boxes touching/crossing it are always kept:
# their geometry is not fully described by the depth map.
NEAR_PLANE = 0.0

_NEG_INF = float("-inf")

# Machine-readable reason markers.  Every decision returns one of these (the
# prefix tells callers whether the object is visible, culled, or just
# unprocessed).
VISIBLE_IN_FRONT = "visible:in_front_of_occluder"
VISIBLE_NEAR_PLANE = "visible:crosses_near_plane"
VISIBLE_DEGENERATE = "visible:degenerate_zero_volume"
VISIBLE_NO_COVERAGE = "visible:no_coverage"
CULLED_OCCLUDED = "culled:occluded"
CULLED_OUTSIDE_FRUSTUM = "culled:outside_frustum"
UNDETERMINED_BUDGET = "undetermined:budget_exhausted"


class DepthPyramid(list):
    """Container for the depth mip chain.

    It is a ``list`` of levels (level 0 first) so it can be indexed and
    iterated directly, while also carrying the source dimensions and the
    per-level ``(width, height)`` table used by the traversal.
    """

    def __init__(self, width, height, levels):
        level_list = list(levels)
        super().__init__(level_list)
        self.width = int(width)
        self.height = int(height)
        self.dims = [(len(level[0]), len(level)) for level in level_list]
        if not level_list:
            raise ValueError("DepthPyramid 至少需要一个层级")
        if self.dims[0] != (self.width, self.height):
            raise ValueError("第 0 级尺寸与 width/height 不一致")

    def levels(self):
        """返回各级深度平面（第 0 级为原始深度图）。"""
        return list(self)


def _validate_depth(depth):
    if not isinstance(depth, (list, tuple)) or not depth:
        raise ValueError("深度图必须是非空二维序列")
    width = len(depth[0])
    if width == 0:
        raise ValueError("深度图宽度必须大于 0")
    grid = []
    for row in depth:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("深度图必须是矩形")
        new_row = []
        for value in row:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("深度值必须是有限数值")
            new_row.append(value)
        grid.append(new_row)
    return grid


def build_pyramid(depth, levels):
    """由深度图构建层级结构。

    第 0 级是原始深度图；第 ``l + 1`` 级的每个单元取第 ``l`` 级对应 2x2
    区域的最大值（最远深度）。每级宽高按 ``ceil(n / 2)`` 向上取整，越界的
    补齐单元视为 ``-inf``，因此补齐永远不会降低最大值。

    实际层级数不会超过到达 1x1 所需的层数：``levels`` 是上限。
    """
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise ValueError("levels 必须是整数")
    if levels < 1:
        raise ValueError("levels 必须大于等于 1")

    grid = _validate_depth(depth)
    height = len(grid)
    width = len(grid[0])
    grids = [grid]

    w, h = width, height
    while len(grids) < levels and (w > 1 or h > 1):
        prev = grids[-1]
        nw, nh = (w + 1) // 2, (h + 1) // 2
        cur = [[_NEG_INF] * nw for _ in range(nh)]
        for y in range(h):
            src = prev[y]
            dst = cur[y // 2]
            for x in range(w):
                value = src[x]
                cx = x // 2
                if value > dst[cx]:
                    dst[cx] = value
        grids.append(cur)
        w, h = nw, nh

    return DepthPyramid(width, height, grids)


def _context(pyramid):
    """Return ``(levels, dims)`` for a DepthPyramid or a plain level list."""
    levels = list(pyramid)
    if not levels:
        raise ValueError("金字塔不能为空")
    dims = getattr(pyramid, "dims", None)
    if dims is None:
        height = len(levels[0])
        width = len(levels[0][0]) if height else 0
        dims = []
        for level in levels:
            if not level:
                dims.append((0, 0))
            else:
                dims.append((len(level[0]), len(level)))
        if dims and dims[0] != (width, height):
            raise ValueError("第 0 级尺寸不一致")
    if len(dims) != len(levels):
        raise ValueError("层级尺寸表与层级数量不一致")
    return levels, dims


def _finite_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s 必须是数值" % name) from None
    if not math.isfinite(number):
        raise ValueError("%s 必须是有限数值" % name)
    return number


def _box_values(box):
    try:
        x = _finite_number(box["x"], "x")
        y = _finite_number(box["y"], "y")
        w = _finite_number(box["w"], "w")
        h = _finite_number(box["h"], "h")
        z = _finite_number(box["z"], "z")
    except (KeyError, TypeError):
        raise ValueError("包围盒需要 x/y/w/h/z 字段") from None
    return x, y, w, h, z


def _pixel_rect(x, y, w, h, width, height):
    """Footprint of a normalized box, as half-open pixel ranges.

    A pixel is included when its half-open cell ``[i/W, (i+1)/W)`` has a
    positive-area intersection with the box.  Boundary-only touching is not
    coverage; this matches the brute-force reference used by the tests.
    """
    px0 = int(math.floor(x * width))
    px1 = int(math.ceil((x + w) * width))
    py0 = int(math.floor(y * height))
    py1 = int(math.ceil((y + h) * height))
    px0 = 0 if px0 < 0 else (width if px0 > width else px0)
    px1 = 0 if px1 < 0 else (width if px1 > width else px1)
    py0 = 0 if py0 < 0 else (height if py0 > height else py0)
    py1 = 0 if py1 < 0 else (height if py1 > height else py1)
    return px0, py0, px1, py1


def _any_at_least(levels, dims, px0, py0, px1, py1, threshold):
    """Is there a covered sample with ``depth >= threshold``?

    Traverses the mip chain from the coarsest level.  A node whose stored
    maximum is below ``threshold`` is pruned wholesale (no sample inside can
    qualify); a fully covered node with a qualifying maximum answers
    immediately.  Only partially covered nodes are descended, so the cost is
    proportional to the boundaries of the footprint, never to its area.
    """
    top = len(levels) - 1
    top_w, top_h = dims[top]
    if top_w <= 0 or top_h <= 0:
        return False

    size = 1 << top
    tx0 = max(0, px0 // size)
    tx1 = min(top_w, (px1 + size - 1) // size)
    ty0 = max(0, py0 // size)
    ty1 = min(top_h, (py1 + size - 1) // size)

    stack = [(top, tx, ty) for ty in range(ty0, ty1) for tx in range(tx0, tx1)]
    while stack:
        level, ix, iy = stack.pop()
        cell = 1 << level
        x0 = ix * cell
        y0 = iy * cell
        x1 = x0 + cell
        y1 = y0 + cell
        # Rectangles are half-open; touching edges carry no area.
        if x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1:
            continue
        if levels[level][iy][ix] < threshold:
            # Farthest sample in this whole subtree is in front of the box.
            continue
        if px0 <= x0 and x1 <= px1 and py0 <= y0 and y1 <= py1:
            return True
        if level == 0:
            # The only level-0 cell overlapping the rectangle is this pixel,
            # and its sample qualifies.
            return True
        child = level - 1
        cw, ch = dims[child]
        cx = ix * 2
        cy = iy * 2
        if cx < cw and cy < ch:
            stack.append((child, cx, cy))
        if cx + 1 < cw and cy < ch:
            stack.append((child, cx + 1, cy))
        if cx < cw and cy + 1 < ch:
            stack.append((child, cx, cy + 1))
        if cx + 1 < cw and cy + 1 < ch:
            stack.append((child, cx + 1, cy + 1))
    return False


def occludee_visible(pyramid, box):
    """判断包围盒是否可见，返回 ``(可见, 原因)``。

    判定顺序（前几项给出确定的边界结论，之后才是遮挡测试）：

    * 零体积（``w <= 0`` 或 ``h <= 0``）：可见，``visible:degenerate_zero_volume``；
    * 完全在视锥外：剔除，``culled:outside_frustum``；
    * 跨近平面（``z <= 0``）：可见，``visible:crosses_near_plane``；
    * 否则在该包围盒覆盖的所有像素的**最远**深度上做保守测试：只要有一个
      覆盖像素的深度不小于包围盒最近深度 ``z`` 就保留，只有全部更远才剔除。
    """
    levels, dims = _context(pyramid)
    width, height = dims[0]
    x, y, w, h, z = _box_values(box)

    # Degenerate (zero-volume) boxes cover no area; keep them, deterministically.
    if w <= 0.0 or h <= 0.0:
        return True, VISIBLE_DEGENERATE

    # Entirely outside the normalized screen rectangle.  This is a frustum
    # decision, not a depth decision; it is definite either way.
    if x + w <= 0.0 or y + h <= 0.0 or x >= 1.0 or y >= 1.0:
        return False, CULLED_OUTSIDE_FRUSTUM

    # Touching or crossing the near plane: the depth map cannot describe the
    # part in front of it, so never cull.
    if z <= NEAR_PLANE:
        return True, VISIBLE_NEAR_PLANE

    px0, py0, px1, py1 = _pixel_rect(x, y, w, h, width, height)
    if px1 <= px0 or py1 <= py0:
        # Cannot happen for a box with positive area intersecting the screen,
        # but keep the conservative answer explicit.
        return True, VISIBLE_NO_COVERAGE

    if _any_at_least(levels, dims, px0, py0, px1, py1, z):
        return True, VISIBLE_IN_FRONT
    return False, CULLED_OCCLUDED


def _priority_key(box, index):
    """Front-to-back order; nearer boxes are resolved first."""
    try:
        z = float(box["z"])
    except (KeyError, TypeError, ValueError):
        z = math.inf
    if z != z:  # NaN would break sorting
        z = math.inf
    return (z, index)


def cull_batch(pyramid, boxes, budget=None):
    """批量判定，返回与输入同序的 ``[(可见, 原因)]``。

    ``budget`` 是最多处理的包围盒数量。超出预算的对象保持
    ``(False, "undetermined:budget_exhausted")``：既没有判定为可见，也没有
    判定为剔除，调用方必须查看原因标记，不能把 False 当作已剔除。处理顺序
    按最近深度优先（从近到远），同深度按输入顺序，保证结果确定。

    ``budget`` 为 ``None`` 或大于等于查询数时，处理全部对象，结果与逐个调用
    :func:`occludee_visible` 完全一致。
    """
    box_list = list(boxes)
    total = len(box_list)
    results = [(False, UNDETERMINED_BUDGET)] * total

    if budget is None or budget >= total:
        order = range(total)
    else:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError("budget 必须是整数或 None")
        if budget <= 0:
            return results
        order = sorted(range(total), key=lambda i: _priority_key(box_list[i], i))
        order = order[:budget]

    visible_fn = occludee_visible
    for index in order:
        results[index] = visible_fn(pyramid, box_list[index])
    return results
