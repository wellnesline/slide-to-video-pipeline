"""rule-based blob orderer (orderer agent の機械的 fallback)。

post の画像を再生成した直後、 blob_anim_prepare で labels.npz / blobs.json は更新できるが、
orderer agent (Claude session 専用) を待たないと order.json が
古いまま → アニメ崩壊する。

このスクリプトは labels.npz / blobs.json から **spatial clustering** で order.json を
機械的に組み立てる。 品質は orderer agent (semantic グルーピング + animation 種別の
意図的選定) には劣るが、 UI 単独で「動くもの」 を出す保証ができる。

方針:
  - blob を y 座標でソート、 overlap または閾値以内なら同 row group
  - 各 group の anim は fade_in 固定 (安全側、 全 5 種から選定は orderer agent の磁気領分)
  - blob_filter は manual:NN,MM 形式
  - shared セクションは skip (shared プールに既に order 完備)

Usage:
  python -m src.post_video.blob_order_rulebased --topic post-NNN [--only-slides 02,03]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import paths
from .slides import load_slides


DEFAULT_ANIM = "fade_in"  # 5 種 (fade_in / wipe_lr / wipe_tb / scale_pop / bounce_in)
TARGET_GROUP_COUNT = 8    # ターゲット group 数 (orderer agent の標準的な分割と整合)
LARGE_BLOB_AREA_RATIO = 0.15  # 全 blob 中 ある blob が 全 area の何 % を占めると「コンテナ」 扱い


def _parse_force_slides(arg: str | None) -> set[str] | None:
    if not arg:
        return None
    out: set[str] = set()
    for token in arg.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            out.add(f"{int(t):02d}")
        except ValueError:
            sys.stderr.write(f"[--only-slides] WARN: invalid sid {t!r}\n")
    return out or None


def cluster_blobs_by_rows(blobs: list[dict], target_count: int = TARGET_GROUP_COUNT) -> list[list[dict]]:
    """blob を centroid y で 概ね target_count 個の group に分割。

    - 大きいコンテナ blob (全 area の 15% 超) は独立 group (各 1 個)
    - 残りを centroid y で sort → N 等分 chunk (target_count - container 数 で分割)

    blob: {id, x0, y0, x1, y1, area}
    """
    if not blobs:
        return []
    total_area = sum(b["area"] for b in blobs)
    threshold = total_area * LARGE_BLOB_AREA_RATIO
    containers = [b for b in blobs if b["area"] >= threshold]
    rest = [b for b in blobs if b["area"] < threshold]

    groups: list[list[dict]] = []
    # コンテナ群 (面積大きい順、 reveal 序盤に出して背景作る)
    for c in sorted(containers, key=lambda b: -b["area"]):
        groups.append([c])

    # 残り blob: centroid y で sort、 概ね chunked groups に
    if rest:
        rest_sorted = sorted(rest, key=lambda b: ((b["y0"] + b["y1"]) / 2, b["x0"]))
        remaining_groups = max(1, target_count - len(groups))
        chunk_size = max(1, (len(rest_sorted) + remaining_groups - 1) // remaining_groups)
        for i in range(0, len(rest_sorted), chunk_size):
            groups.append(rest_sorted[i:i + chunk_size])
    return groups


def build_order_for_slide(blobs_json: dict, diagram_id: str) -> dict:
    """blobs.json から rule-based order.json を組み立て。"""
    blobs = blobs_json.get("blobs", [])
    groups = cluster_blobs_by_rows(blobs)
    order_groups: list[dict] = []
    for i, group_blobs in enumerate(groups):
        # group 内は x 座標でソート (左→右)
        group_blobs_sorted = sorted(group_blobs, key=lambda b: b["x0"])
        ids = [b["id"] for b in group_blobs_sorted]
        order_groups.append({
            "label": f"row_{i+1:02d}",
            "anim": DEFAULT_ANIM,
            "blob_filter": "manual:" + ",".join(ids),
            "resolved_blobs": [
                {k: b[k] for k in ("id", "x0", "y0", "x1", "y1", "area")}
                for b in group_blobs_sorted
            ],
        })
    return {
        "diagram_id": diagram_id,
        "groups": order_groups,
        "_generated_by": "blob_order_rulebased",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--only-slides", default=None,
                    help="カンマ区切り sid。 指定 slide のみ生成。 default: post 個別 全 slide")
    ap.add_argument("--overwrite", action="store_true",
                    help="既存 order.json も上書き (default: 既存あればスキップ)")
    args = ap.parse_args(argv)

    topic = args.topic
    orders_dir = paths.orders_dir(topic)
    blobs_dir = paths.blobs_dir(topic)
    if not blobs_dir.exists():
        sys.stderr.write(f"blobs dir 不在: {blobs_dir}\n  先に blob_anim_prepare を実行してください\n")
        return 1
    orders_dir.mkdir(parents=True, exist_ok=True)

    slides = load_slides(topic)
    targets = [s for s in slides if not s["shared_id"]]  # shared は skip
    only = _parse_force_slides(args.only_slides)
    if only:
        before = len(targets)
        targets = [s for s in targets if s["id"] in only]
        print(f"[--only-slides] {sorted(only)} のみ ({before} → {len(targets)})")

    n_written = 0
    n_skipped = 0
    for s in targets:
        sid = s["id"]
        blob_json_path = blobs_dir / f"slide_{sid}" / "blobs.json"
        if not blob_json_path.exists():
            print(f"  [skip] {sid}: blobs.json 不在 (先に blob_anim_prepare 必要)")
            continue
        order_path = orders_dir / f"slide_{sid}_order.json"
        if order_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        try:
            bdata = json.loads(blob_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {sid}: blobs.json parse error: {e}")
            continue
        order = build_order_for_slide(bdata, f"slide_{sid}")
        order_path.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [ok] {sid}: {len(order['groups'])} groups, {sum(len(g['resolved_blobs']) for g in order['groups'])} blobs")
        n_written += 1

    print(f"\n=== done: {n_written} written, {n_skipped} skipped (existing) ===")
    if n_written > 0:
        print("注意: rule-based なので品質は orderer agent (Claude session) に劣ります。")
        print("      品質改善するなら UI 「orderer agent で改善」 ボタンから依頼してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
