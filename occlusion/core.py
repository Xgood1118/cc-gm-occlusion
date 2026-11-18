# -*- coding: utf-8 -*-
"""occlusion"""

from __future__ import annotations

__all__ = ["DepthPyramid", "build_pyramid", "cull_batch", "occludee_visible"]


class DepthPyramid:
    """DepthPyramid"""

    def __init__(self, width, height, levels):
        """层级深度金字塔容器"""
        raise NotImplementedError("DepthPyramid.__init__ 尚未实现")

    def levels(self):
        """返回各级深度平面"""
        raise NotImplementedError("DepthPyramid.levels 尚未实现")


def build_pyramid(depth, levels):
    """由深度图构建层级结构"""
    raise NotImplementedError("build_pyramid 尚未实现")


def occludee_visible(pyramid, box):
    """判断包围盒是否可见，返回 (bool, 原因)"""
    raise NotImplementedError("occludee_visible 尚未实现")


def cull_batch(pyramid, boxes, budget=None):
    """批量判定，返回 [(可见, 原因)]"""
    raise NotImplementedError("cull_batch 尚未实现")
