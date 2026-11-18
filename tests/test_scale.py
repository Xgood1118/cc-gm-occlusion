# -*- coding: utf-8 -*-
"""Scale and budget tests: the culler must not degenerate into per-pixel work."""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import occlusion as mod  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_hierarchy import make_boxes  # noqa: E402

BUDGET_SECONDS = 3.0


def timed_depth(width, height, seed=11):
    rng = random.Random(seed)
    return [[0.15 + 0.85 * rng.random() for _ in range(width)] for _ in range(height)]


def levels_for(width, height):
    return max(1, max(width, height).bit_length())


def test_512_2000_boxes_under_three_seconds():
    width = height = 512
    depth = timed_depth(width, height)
    p = mod.build_pyramid(depth, levels_for(width, height))
    boxes = make_boxes(2000, seed=21)
    start = time.perf_counter()
    out = mod.cull_batch(p, boxes)
    elapsed = time.perf_counter() - start
    assert len(out) == 2000
    assert elapsed < BUDGET_SECONDS, "单帧剔除耗时 %.3fs" % elapsed


def test_doubling_resolution_is_sublinear():
    """Cost of a box query depends on footprint, not on image area."""
    boxes = make_boxes(300, seed=33)
    times = {}
    for width, height in ((256, 256), (512, 512), (1024, 1024)):
        depth = timed_depth(width, height)
        p = mod.build_pyramid(depth, levels_for(width, height))
        mod.cull_batch(p, boxes)  # warm up caches
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            mod.cull_batch(p, boxes)
            best = min(best, time.perf_counter() - start)
        times[width] = best
    # Area grows 4x per step; a per-pixel scan would grow ~4x.  Allow ample
    # noise headroom but require clearly sublinear growth.
    growth = times[1024] / max(times[256], 1e-9)
    assert growth < 4.0, times
    assert times[512] / max(times[256], 1e-9) < 4.0, times
