# -*- coding: utf-8 -*-
"""Correctness tests for the conservative hierarchy and the culler."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import occlusion as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# Independent brute-force reference.  Intentionally shares no code with
# occlusion.core so a bug in the footprint maths cannot hide in both.
# ---------------------------------------------------------------------------
def brute_visible(depth, box):
    height = len(depth)
    width = len(depth[0])
    x, y, w, h, z = box["x"], box["y"], box["w"], box["h"], box["z"]
    if w <= 0.0 or h <= 0.0:
        return True
    if x + w <= 0.0 or y + h <= 0.0 or x >= 1.0 or y >= 1.0:
        return False
    if z <= 0.0:
        return True
    px0 = max(0, min(width, int(math.floor(x * width))))
    px1 = max(0, min(width, int(math.ceil((x + w) * width))))
    py0 = max(0, min(height, int(math.floor(y * height))))
    py1 = max(0, min(height, int(math.ceil((y + h) * height))))
    for py in range(py0, py1):
        for px in range(px0, px1):
            if depth[py][px] >= z:
                return True
    return False


def brute_min_depth_under_box(depth, box):
    """Minimum depth over the covered pixels (the most favourable sample)."""
    height = len(depth)
    width = len(depth[0])
    px0 = max(0, min(width, int(math.floor(box["x"] * width))))
    px1 = max(0, min(width, int(math.ceil((box["x"] + box["w"]) * width))))
    py0 = max(0, min(height, int(math.floor(box["y"] * height))))
    py1 = max(0, min(height, int(math.ceil((box["y"] + box["h"]) * height))))
    values = [depth[py][px] for py in range(py0, py1) for px in range(px0, px1)]
    return min(values) if values else None


def make_depth(width, height, seed=1234):
    rng = random.Random(seed)
    return [[rng.uniform(0.0, 1.0) for _ in range(width)] for _ in range(height)]


def make_boxes(count, seed=99):
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        w = rng.choice([0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 1.5])
        h = rng.choice([0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 1.5])
        x = rng.uniform(-0.5, 1.0)
        y = rng.uniform(-0.5, 1.0)
        z = rng.choice([-0.5, 0.0, 1e-9, 0.05, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0])
        boxes.append({"x": x, "y": y, "w": w, "h": h, "z": z})
    return boxes


def levels_for(width, height):
    return max(1, max(width, height).bit_length())


# ---------------------------------------------------------------------------
# 1. Hierarchy: every level stores the farthest sample of its real footprint.
# ---------------------------------------------------------------------------
def test_pyramid_levels_are_region_maxima():
    width, height = 13, 7  # deliberately not powers of two, both odd
    depth = make_depth(width, height)
    levels = levels_for(width, height)
    p = mod.build_pyramid(depth, levels)
    assert len(p) == levels
    for lvl, plane in enumerate(p):
        cell = 1 << lvl
        assert len(plane) == (height + cell - 1) // cell
        assert len(plane[0]) == (width + cell - 1) // cell
        for iy, row in enumerate(plane):
            for ix, value in enumerate(row):
                x0, y0 = ix * cell, iy * cell
                x1, y1 = min(width, x0 + cell), min(height, y0 + cell)
                assert x1 > x0 and y1 > y0  # every cell covers real pixels
                real_max = max(depth[y][x] for y in range(y0, y1)
                               for x in range(x0, x1))
                assert value >= real_max - 1e-12, (lvl, ix, iy)
                assert value <= real_max + 1e-12, (lvl, ix, iy)


def test_pyramid_monotone_and_root_is_global_max():
    width, height = 11, 6
    depth = make_depth(width, height, seed=7)
    p = mod.build_pyramid(depth, 32)
    assert len(p[-1]) == 1 and len(p[-1][0]) == 1
    assert p[-1][0][0] >= max(max(row) for row in depth) - 1e-12
    for lvl in range(1, len(p)):
        child = p[lvl - 1]
        for iy, row in enumerate(p[lvl]):
            for ix, value in enumerate(row):
                for cy in (2 * iy, 2 * iy + 1):
                    for cx in (2 * ix, 2 * ix + 1):
                        if cy < len(child) and cx < len(child[0]):
                            assert value >= child[cy][cx] - 1e-12


def test_pyramid_size_one_reaches_one_level():
    p = mod.build_pyramid([[0.4]], 5)
    assert len(p) == 1
    assert p.levels() == [p[0]]


def test_pyramid_rejects_bad_input():
    for bad_levels in (0, -1, 2.5, True):
        try:
            mod.build_pyramid([[0.1]], bad_levels)
        except ValueError:
            pass
        else:
            raise AssertionError("levels=%r 应被拒绝" % (bad_levels,))
    for bad_depth in ([], [[]], [[0.1], [0.2, 0.3]], [[float("nan")]], [[float("inf")]]):
        try:
            mod.build_pyramid(bad_depth, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("depth=%r 应被拒绝" % (bad_depth,))
    p = mod.build_pyramid([[0.5, 0.5], [0.5, 0.5]], 2)
    try:
        p.levels()  # noqa: B018 - must not raise
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError("levels() 不应抛异常") from exc


# ---------------------------------------------------------------------------
# 2. Zero false culls against brute force, on several grid shapes.
# ---------------------------------------------------------------------------
def test_zero_false_culls_against_bruteforce():
    shapes = [(4, 4), (13, 7), (64, 64), (33, 17)]
    for width, height in shapes:
        depth = make_depth(width, height, seed=width * 100 + height)
        p = mod.build_pyramid(depth, levels_for(width, height))
        boxes = make_boxes(200, seed=width)
        results = mod.cull_batch(p, boxes)
        assert len(results) == len(boxes)
        false_culls = 0
        false_keeps = 0
        for box, (ok, why) in zip(boxes, results):
            reference = brute_visible(depth, box)
            if not ok and why.startswith("culled:") and reference:
                false_culls += 1
            if ok and why.startswith("visible:") and not reference:
                false_keeps += 1
            assert isinstance(ok, bool) and isinstance(why, str) and why
        assert false_culls == 0, (width, height)
        # The culler is also exact on this data: no conservative over-keeps
        # outside the explicit boundary cases.
        assert false_keeps == 0, (width, height)


def test_equality_with_occluder_depth_is_visible():
    depth = [[0.3, 0.3], [0.3, 0.3]]
    p = mod.build_pyramid(depth, 2)
    ok, why = mod.occludee_visible(p, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.3})
    assert ok is True and why == mod.VISIBLE_IN_FRONT
    ok, why = mod.occludee_visible(
        p, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": math.nextafter(0.3, 1.0)}
    )
    assert ok is False and why == mod.CULLED_OCCLUDED


def test_cull_requires_every_covered_sample_farther():
    boxes = make_boxes(120, seed=5)
    depth = make_depth(9, 6, seed=5)
    p = mod.build_pyramid(depth, levels_for(9, 6))
    for box in boxes:
        ok, why = mod.occludee_visible(p, box)
        if why.startswith("culled:occluded"):
            assert brute_min_depth_under_box(depth, box) < box["z"]


# ---------------------------------------------------------------------------
# 3. Batch equals per-box, and budget marks work as undetermined.
# ---------------------------------------------------------------------------
def test_batch_equals_individual():
    depth = make_depth(17, 11, seed=3)
    p = mod.build_pyramid(depth, levels_for(17, 11))
    boxes = make_boxes(150, seed=4)
    batch = mod.cull_batch(p, boxes)
    single = [mod.occludee_visible(p, box) for box in boxes]
    assert batch == single
    assert mod.cull_batch(p, boxes, budget=len(boxes)) == single
    assert mod.cull_batch(p, boxes, budget=None) == single
    assert mod.cull_batch(p, tuple(boxes)) == single


def test_budget_processes_by_priority_and_marks_the_rest():
    depth = [[0.99] * 4 for _ in range(4)]
    p = mod.build_pyramid(depth, 2)
    boxes = [
        {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1, "z": 0.9},
        {"x": 0.2, "y": 0.0, "w": 0.1, "h": 0.1, "z": 0.1},
        {"x": 0.4, "y": 0.0, "w": 0.1, "h": 0.1, "z": 0.5},
    ]
    out = mod.cull_batch(p, boxes, budget=1)
    assert out[1] == mod.occludee_visible(p, boxes[1])
    assert out[0] == (False, mod.UNDETERMINED_BUDGET)
    assert out[2] == (False, mod.UNDETERMINED_BUDGET)
    # zero budget: nothing decided, nothing silently culled
    assert mod.cull_batch(p, boxes, budget=0) == [(False, mod.UNDETERMINED_BUDGET)] * 3
    # huge budget behaves like no budget
    assert mod.cull_batch(p, boxes, budget=1000) == mod.cull_batch(p, boxes)


# ---------------------------------------------------------------------------
# 4. Boundary cases: outside frustum, near plane, degenerate boxes.
# ---------------------------------------------------------------------------
def test_outside_frustum_is_culled_deterministically():
    p = mod.build_pyramid([[0.5]], 1)
    cases = [
        {"x": -3.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.9},
        {"x": 0.0, "y": -2.0, "w": 1.0, "h": 1.5, "z": 0.9},
        {"x": 1.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.9},
        {"x": 0.0, "y": 1.5, "w": 1.0, "h": 1.0, "z": 0.9},
    ]
    for box in cases:
        ok, why = mod.occludee_visible(p, box)
        assert ok is False and why == mod.CULLED_OUTSIDE_FRUSTUM
        assert mod.cull_batch(p, [box])[0] == (False, mod.CULLED_OUTSIDE_FRUSTUM)


def test_crossing_near_plane_is_never_culled():
    p = mod.build_pyramid([[0.01]], 1)
    for z in (-1.0, 0.0, -1e-12):
        ok, why = mod.occludee_visible(p, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": z})
        assert ok is True and why == mod.VISIBLE_NEAR_PLANE


def test_degenerate_boxes_are_visible_and_labelled():
    p = mod.build_pyramid([[0.01, 0.01], [0.01, 0.01]], 2)
    cases = [
        {"x": 0.0, "y": 0.0, "w": 0.0, "h": 1.0, "z": 0.9},
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.0, "z": 0.9},
        {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "z": 0.9},
        {"x": 0.0, "y": 0.0, "w": -0.2, "h": 0.5, "z": 0.9},
    ]
    for box in cases:
        ok, why = mod.occludee_visible(p, box)
        assert ok is True and why == mod.VISIBLE_DEGENERATE
        assert brute_visible([[0.01, 0.01], [0.01, 0.01]], box) is True


def test_box_forcing_near_plane_ahead_of_occluders():
    # The most adverse sample is the farthest one; a box must not be culled
    # even when most covered pixels are much farther away.
    depth = [[0.2, 0.9], [0.9, 0.9]]
    p = mod.build_pyramid(depth, 2)
    ok, why = mod.occludee_visible(p, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.5})
    assert ok is True and why == mod.VISIBLE_IN_FRONT


def test_invalid_boxes_rejected():
    p = mod.build_pyramid([[0.5]], 1)
    for box in ({"w": 1.0, "h": 1.0, "z": 0.5},
                {"x": float("nan"), "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.5},
                {"x": 0.0, "y": 0.0, "w": "a", "h": 1.0, "z": 0.5}):
        try:
            mod.occludee_visible(p, box)
        except ValueError:
            pass
        else:
            raise AssertionError("非法包围盒应被拒绝: %r" % (box,))
