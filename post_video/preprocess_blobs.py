"""blob bbox JSON に空間グラフ情報を機械付与。

verifier 指摘の論点7-(a)(c) を実装:
  (a) neighbors — 各 blob の bbox 中心距離 < 200px の他 blob ID リスト
  (c) contained_by — この blob を bbox 70%以上 内包する 大 blob ID リスト

機械で前処理して LLM (orderer) に渡すことで、 LLM の mapping負荷を
構造的に下げる (= LLM ブレが入らない決定的層で前処理)。

Usage:
  python tools/anim_quality/preprocess_blobs.py --topic <post-id> --slides 05
  python tools/anim_quality/preprocess_blobs.py --topic <post-id>

入力: <topic>/work/blobs/slide_<NN>/blobs.json
出力: 同 path に上書き (各 blob に neighbors, contained_by フィールド追加)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _bbox_center(b: dict) -> tuple[float, float]:
    return ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2)


def _bbox_area(b: dict) -> int:
    return max(0, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))


def _overlap_ratio(outer: dict, inner: dict) -> float:
    """outer bbox が inner bbox を何 % 内包するか (inner視点)。"""
    ix0 = max(outer["x0"], inner["x0"])
    iy0 = max(outer["y0"], inner["y0"])
    ix1 = min(outer["x1"], inner["x1"])
    iy1 = min(outer["y1"], inner["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    inner_area = _bbox_area(inner)
    return inter / inner_area if inner_area > 0 else 0.0


def preprocess(blobs: list[dict],
               neighbor_max_dist: int = 250,
               contain_overlap_threshold: float = 0.7,
               contain_min_outer_area: int = 5000,
               row_y_tol: int = 25) -> list[dict]:
    """各 blob に neighbors と contained_by を付与。 元データは破壊しない。"""
    out = [dict(b) for b in blobs]
    centers = {b["id"]: _bbox_center(b) for b in out}
    areas = {b["id"]: _bbox_area(b) for b in out}

    for b in out:
        bid = b["id"]
        bcx, bcy = centers[bid]
        b_h = b["y1"] - b["y0"]
        # neighbors (中心距離ベース)
        nbrs = []
        for other in out:
            oid = other["id"]
            if oid == bid:
                continue
            ocx, ocy = centers[oid]
            dist = ((bcx - ocx) ** 2 + (bcy - ocy) ** 2) ** 0.5
            if dist <= neighbor_max_dist:
                nbrs.append({"id": oid, "dist": round(dist, 1)})
        # 距離順にソート、 上位10件まで
        nbrs.sort(key=lambda n: n["dist"])
        b["neighbors"] = [n["id"] for n in nbrs[:10]]

        # row_neighbors: 「同じ行」 にいる blob (y中心が ±row_y_tol、 高さが2倍以内、 水平 gap < 100px)
        row_mates = []
        for other in out:
            oid = other["id"]
            if oid == bid:
                continue
            ocx, ocy = centers[oid]
            o_h = other["y1"] - other["y0"]
            if abs(bcy - ocy) > row_y_tol:
                continue
            if max(b_h, o_h) > min(b_h, o_h) * 2.0:
                continue
            if b["x1"] < other["x0"]:
                gap = other["x0"] - b["x1"]
            elif other["x1"] < b["x0"]:
                gap = b["x0"] - other["x1"]
            else:
                gap = 0
            if gap <= 100:
                row_mates.append({"id": oid, "x0": other["x0"]})
        # 左→右 順
        row_mates.sort(key=lambda r: r["x0"])
        b["row_neighbors"] = [r["id"] for r in row_mates]

        # contained_by (この blob を 内包する大 blob ID リスト)
        contained_by = []
        for outer in out:
            oid = outer["id"]
            if oid == bid:
                continue
            if areas[oid] < contain_min_outer_area:
                continue
            if areas[oid] < areas[bid] * 1.5:
                continue  # 大きさが近いなら 内包扱いしない
            if _overlap_ratio(outer, b) >= contain_overlap_threshold:
                contained_by.append(oid)
        b["contained_by"] = contained_by

    return out


def _topic_dir(topic: str) -> Path:
    # POST_VIDEO_ROOT / POST_VIDEO_POSTS_SUBDIR 切替に追随するため paths.topic_dir に委譲する。
    # (出力先ルートを POST_VIDEO_ROOT で切り替えたときに、ハードコードされたパスだと topic が見つからなくなる問題の修正)
    try:
        from . import paths as _paths
        d = _paths.topic_dir(topic)
        if d.exists():
            return d
    except Exception:
        pass
    # フォールバック: 従来のハードコード候補 (paths 未解決時の保険)
    cands = [
        ROOT / "content_pipeline" / "posts" / topic,
        ROOT / "content_pipeline" / "alt_posts" / topic,
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"topic dir 不在: {topic}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--slides", default=None)
    ap.add_argument("--neighbor-max-dist", type=int, default=200)
    ap.add_argument("--contain-threshold", type=float, default=0.7)
    args = ap.parse_args(argv)

    topic_dir = _topic_dir(args.topic)
    blobs_root = topic_dir / "work" / "blobs"
    only = [f"{int(s):02d}" for s in args.slides.split(",")] if args.slides else None

    targets = sorted(blobs_root.glob("slide_*/blobs.json"))
    print(f"=== preprocess_blobs ({args.topic}, {len(targets)} slides) ===")
    n_processed = 0
    for f in targets:
        sid = f.parent.name.replace("slide_", "")
        if only and sid not in only:
            continue
        data = json.loads(f.read_text())
        blobs = data.get("blobs", [])
        new_blobs = preprocess(blobs, args.neighbor_max_dist, args.contain_threshold)
        # bak保存 (初回のみ)
        bak = f.with_suffix(".json.preprocess_bak")
        if not bak.exists():
            bak.write_text(f.read_text())
        data["blobs"] = new_blobs
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        # サマリ
        with_nbrs = sum(1 for b in new_blobs if b.get("neighbors"))
        with_contained = sum(1 for b in new_blobs if b.get("contained_by"))
        print(f"  slide_{sid}: {len(new_blobs)} blobs, neighbors付き={with_nbrs}, contained_by付き={with_contained}")
        n_processed += 1
    print(f"=== done: {n_processed} slides 処理 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
