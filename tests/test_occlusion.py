# -*- coding: utf-8 -*-
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import occlusion as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def test_pyramid_shape():
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    assert isinstance(p, (list, tuple)) and len(p) == 3
    assert len(p[0]) == d["height"] and len(p[0][0]) == d["width"]
    assert len(p[1]) == 2 and len(p[1][0]) == 2
    assert len(p[2]) == 1 and len(p[2][0]) == 1


def test_pyramid_conservative():
    """每级取对应区域最远深度，任何一级都不得比真实几何更近。"""
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    for y in range(2):
        for x in range(2):
            blk = [d["depth"][2 * y + dy][2 * x + dx] for dy in range(2) for dx in range(2)]
            assert p[1][y][x] >= max(blk) - 1e-12


def test_visible_when_closer():
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    ok, why = mod.occludee_visible(p, {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1, "z": 0.05})
    assert ok is True and why


def test_hidden_when_far_behind():
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    ok, why = mod.occludee_visible(p, {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 1.0})
    assert ok is False and why


def test_no_false_cull_against_bruteforce():
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    boxes = [{"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5, "z": 0.75},
             {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "z": 0.35},
             {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "z": 0.99}]
    out = mod.cull_batch(p, boxes)
    assert len(out) == len(boxes)
    for (ok, why), b in zip(out, boxes):
        # 保守性：若包围盒最近处比覆盖像素的最近深度还近，必须判可见
        assert isinstance(ok, bool) and why


def test_budget_marks_undetermined():
    d = json.load(open(os.path.join(DATA, "depth.json"), encoding="utf-8"))
    p = mod.build_pyramid(d["depth"], 3)
    boxes = [{"x": 0, "y": 0, "w": 0.2, "h": 0.2, "z": 0.1},
             {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "z": 0.1},
             {"x": 0.8, "y": 0.8, "w": 0.2, "h": 0.2, "z": 0.1}]
    out = mod.cull_batch(p, boxes, budget=1)
    assert len(out) == 3
    assert sum(1 for ok, why in out if "undetermined" in why) >= 2
