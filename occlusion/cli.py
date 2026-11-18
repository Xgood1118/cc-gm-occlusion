# -*- coding: utf-8 -*-
"""Command line driver for the hierarchical occlusion culler.

``python -m occlusion.cli data/depth.json --out cull.json`` builds a full mip
chain from the depth image, generates a deterministic box set, culls it, and
cross-checks every culled box against an independent per-pixel brute force
scan.  The report is written to ``--out`` and summarised on stdout.

``--bench`` additionally times two workloads (512x512 / 2000 boxes and
1024x1024 / 2000 boxes) and compares them with a per-pixel reference.
"""

import argparse
import json
import math
import random
import time

import occlusion as m

# Default box count for the report run.  Matches the workload in the brief.
DEFAULT_BOXES = 2000

BENCH_SCALES = (("512x512", 512, 512), ("1024x1024", 1024, 1024))
BENCH_BOXES = 2000
BENCH_SEED = 20240601


def levels_for(width, height):
    """Levels needed for a complete chain down to a single cell."""
    return max(1, max(width, height).bit_length())


def _pixel_rect(box, width, height):
    """Independent footprint computation (half-open pixel cells)."""
    x, y = box["x"], box["y"]
    w, h = box["w"], box["h"]
    px0 = int(math.floor(x * width))
    px1 = int(math.ceil((x + w) * width))
    py0 = int(math.floor(y * height))
    py1 = int(math.ceil((y + h) * height))
    px0 = 0 if px0 < 0 else (width if px0 > width else px0)
    px1 = 0 if px1 < 0 else (width if px1 > width else px1)
    py0 = 0 if py0 < 0 else (height if py0 > height else py0)
    py1 = 0 if py1 < 0 else (height if py1 > height else py1)
    return px0, py0, px1, py1


def bruteforce_visible(depth, box, width, height):
    """Per-pixel reference implementation, used only to audit culling."""
    x, y, w, h, z = box["x"], box["y"], box["w"], box["h"], box["z"]
    if w <= 0.0 or h <= 0.0:
        return True
    if x + w <= 0.0 or y + h <= 0.0 or x >= 1.0 or y >= 1.0:
        return False
    if z <= 0.0:
        return True
    px0, py0, px1, py1 = _pixel_rect(box, width, height)
    for py in range(py0, py1):
        row = depth[py]
        for px in range(px0, px1):
            if row[px] >= z:
                return True
    return False


def gen_boxes(count, seed):
    """Deterministic pseudo-random box set (LCG so it is reproducible)."""
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        w = 0.01 * math.exp(rng.random() * math.log(100.0))  # 0.01 .. 1.0
        h = 0.01 * math.exp(rng.random() * math.log(100.0))
        w = min(w, 1.0)
        h = min(h, 1.0)
        x = rng.random() * (1.0 - w)
        y = rng.random() * (1.0 - h)
        z = rng.uniform(0.0, 1.05)
        boxes.append({"x": x, "y": y, "w": w, "h": h, "z": z})
    return boxes


def _noise_grid(width, height, cells, rng):
    """Bilinear value noise on a coarse lattice, returned at full resolution."""
    points = [[rng.random() for _ in range(cells + 1)] for _ in range(cells + 1)]
    out = []
    for y in range(height):
        fy = y * cells / max(1, height - 1) if height > 1 else 0.0
        y0 = min(cells - 1, int(fy))
        ty = fy - y0
        row_a = points[y0]
        row_b = points[y0 + 1]
        row = []
        for x in range(width):
            fx = x * cells / max(1, width - 1) if width > 1 else 0.0
            x0 = min(cells - 1, int(fx))
            tx = fx - x0
            top = row_a[x0] * (1 - tx) + row_a[x0 + 1] * tx
            bot = row_b[x0] * (1 - tx) + row_b[x0 + 1] * tx
            row.append(top * (1 - ty) + bot * ty)
        out.append(row)
    return out


def gen_depth(width, height, seed):
    """Deterministic depth image resembling layered geometry, in [0, 1].

    Several octaves of value noise, stretched toward the far plane, plus a
    constant near slab on the left so there is real occlusion to find.
    """
    rng = random.Random(seed)
    octaves = []
    total = [0.0] * (width * height)
    weight = 1.0
    norm = 0.0
    for cells in (2, 4, 8, 16):
        octave = _noise_grid(width, height, cells, rng)
        for i, (a, b) in enumerate(zip(total, (v for row in octave for v in row))):
            total[i] = a + weight * b
        norm += weight
        weight *= 0.55
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            v = total[y * width + x] / norm
            d = 0.15 + 0.85 * v
            if x < width // 8:  # near slab covering the left edge
                d = 0.1 + 0.1 * d
            row.append(min(1.0, d))
        grid.append(row)
    return grid


def summarize(pyramid, depth, width, height, boxes, budget=None):
    started = time.perf_counter()
    results = m.cull_batch(pyramid, boxes, budget=budget)
    elapsed = time.perf_counter() - started

    counts = {"visible": 0, "culled": 0, "undetermined": 0}
    false_culls = 0
    for box, (ok, why) in zip(boxes, results):
        if why.startswith("undetermined:"):
            counts["undetermined"] += 1
            continue
        if ok:
            counts["visible"] += 1
            continue
        counts["culled"] += 1
        if bruteforce_visible(depth, box, width, height):
            false_culls += 1

    return {
        "levels": len(pyramid),
        "queries": len(boxes),
        "budget": budget,
        "visible": counts["visible"],
        "culled": counts["culled"],
        "undetermined": counts["undetermined"],
        "cull_rate": (counts["culled"] / len(boxes)) if boxes else 0.0,
        "false_culls": false_culls,
        "cull_seconds": elapsed,
        "results": [[bool(ok), why] for ok, why in results],
    }


def run_bench():
    """Time the culler at two scales and against a per-pixel reference."""
    rows = []
    for label, width, height in BENCH_SCALES:
        depth = gen_depth(width, height, BENCH_SEED)
        boxes = gen_boxes(BENCH_BOXES, BENCH_SEED + 1)

        build_start = time.perf_counter()
        pyramid = m.build_pyramid(depth, levels_for(width, height))
        build_seconds = time.perf_counter() - build_start

        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            results = m.cull_batch(pyramid, boxes)
            best = min(best, time.perf_counter() - start)
        culled = sum(1 for ok, why in results if not ok and why.startswith("culled:"))
        false_culls = sum(
            1
            for box, (ok, why) in zip(boxes, results)
            if not ok and why.startswith("culled:")
            and bruteforce_visible(depth, box, width, height)
        )

        # Per-pixel reference over a subset (full 2000 would dominate runtime).
        sample = boxes[:100]
        start = time.perf_counter()
        for box in sample:
            bruteforce_visible(depth, box, width, height)
        brute_per_box = (time.perf_counter() - start) / len(sample)

        rows.append({
            "scale": label,
            "width": width,
            "height": height,
            "levels": len(pyramid),
            "build_seconds": build_seconds,
            "cull_seconds": best,
            "cull_rate": culled / len(boxes),
            "false_culls": false_culls,
            "bruteforce_ms_per_box": brute_per_box * 1000.0,
            "bruteforce_full_estimate_seconds": brute_per_box * len(boxes),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="occlusion")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boxes", type=int, default=DEFAULT_BOXES)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--bench", action="store_true",
                    help="also run the two-scale timing benchmark")
    a = ap.parse_args()

    with open(a.path, encoding="utf-8") as fh:
        d = json.load(fh)
    width, height = int(d["width"]), int(d["height"])
    depth = d["depth"]
    if len(depth) != height or any(len(row) != width for row in depth):
        raise SystemExit("depth.json 的 width/height 与 depth 不一致")

    pyramid = m.build_pyramid(depth, levels_for(width, height))
    boxes = gen_boxes(a.boxes, a.seed)
    report = summarize(pyramid, depth, width, height, boxes, budget=a.budget)

    if a.bench:
        report["benchmark"] = run_bench()

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    summary = {k: v for k, v in report.items() if k != "results"}
    if "benchmark" in summary:
        bench = summary.pop("benchmark")
        summary["benchmark"] = [
            {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}
            for row in bench
        ]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
