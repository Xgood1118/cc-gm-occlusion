# -*- coding: utf-8 -*-
import argparse
import json
import occlusion as m


def main():
    ap = argparse.ArgumentParser(description="occlusion")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    p = m.build_pyramid(d["depth"], 3)
    print(json.dumps({"levels": len(p)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
