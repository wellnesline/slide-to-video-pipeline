"""order JSON post-process — subagent の merge ミスをルールベースで補正。

verifier の指摘  を反映した v2 設計:
  - gap default 48px (80px は --aggressive で opt-in)
  - container keyword guard (frame/box/container/枠/箱/bubble label は merge 除外)
  - group 単位判定 (pair 単位ではなく、 group ペア全体の代表性で判定)
  - merge 先候補が複数 group に跨る場合は WARN で人間 escalation

ルール (適用順):
  R1. 重複blob除去 — 同じ blob ID が複数 group に登場時、 最初の group のみに残す
  R2. 枠内包 merge — blob A が blob B を 70% 以上 bbox 内包 + area_A > 5000 のとき同一 group
                     ただし両方が container keyword group なら別構造として skip
  R3. 水平方向近接 merge — gap < 48px + y中心 ±25px + 高さ2倍以内 の隣接 blob ペア
                          ただし:
                          - container keyword group は除外
                          - 同 anim 型かつ label prefix が異なる場合 skip (意図的演出尊重)
                          - 跨ぎ group が 3 以上なら skip + WARN
  R4. 取りこぼし blob を末尾 fade_in 追加 — order に無い blob (area > 1000) を末尾の fade_in group に追加
  R5. 機械的ラベル正規化 — row_NN / col_NN / group_NN を merged_g{NN} に置換

Usage (CLI):
  python -m post_video.order_merge \\
    --topic <post-id> \\
    [--slides 14,16] \\
    [--gap 48] [--aggressive] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

CONTAINER_KEYWORDS = [
    "frame", "box", "container", "枠", "箱", "bubble", "outer",
    "speech_bubble", "card", "panel",
]
CONTAINER_EXCLUDE_KEYWORDS = [
    "checkbox", "textbox",  # box キーワードでも 「checkbox」 等は要素扱い
]
CONTAINER_MIN_AREA = 10000

MECHANICAL_LABEL_RE = re.compile(r"\b(row|col|group|block)_\d+\b", re.IGNORECASE)


def _is_container_group(g: dict, blob_meta: dict[str, dict] | None = None) -> bool:
    """label keyword + bbox area の両条件で判定 (verifier 指摘 axis_2 抜け穴対応)。

    label に container kw が含まれるが除外 kw 「checkbox」 等にもマッチする場合は False。
    keyword判定通過後、 blob_meta が与えられたら代表 blob area が CONTAINER_MIN_AREA 以上か確認。
    """
    label = (g.get("label") or "").lower()
    if any(kw in label for kw in CONTAINER_EXCLUDE_KEYWORDS):
        return False
    if not any(kw in label for kw in CONTAINER_KEYWORDS):
        return False
    if blob_meta is None:
        return True
    ids = _group_blob_ids(g)
    if not ids:
        return False
    candidates = [blob_meta[bid] for bid in ids if bid in blob_meta]
    if not candidates:
        return False
    max_area = max(_bbox_area(b) for b in candidates)
    return max_area >= CONTAINER_MIN_AREA


def _group_blob_ids(g: dict) -> list[str]:
    out: list[str] = []
    for sub in g.get("resolved_blobs", []):
        for b in sub:
            out.append(b["id"])
    return out


def _blob_meta_index(blobs: list[dict]) -> dict[str, dict]:
    return {b["id"]: b for b in blobs}


def _bbox_overlap_ratio(a: dict, b: dict) -> float:
    """a bbox が b bbox を どれだけ内包するか (b 視点での被覆率)。"""
    ix0 = max(a["x0"], b["x0"])
    iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"])
    iy1 = min(a["y1"], b["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    b_area = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / b_area if b_area > 0 else 0.0


def _bbox_area(b: dict) -> int:
    return (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])


def _horizontal_neighbor(a: dict, b: dict, gap_max: int, y_tol: int) -> bool:
    cy_a = (a["y0"] + a["y1"]) / 2
    cy_b = (b["y0"] + b["y1"]) / 2
    if abs(cy_a - cy_b) > y_tol:
        return False
    h_a = a["y1"] - a["y0"]
    h_b = b["y1"] - b["y0"]
    if max(h_a, h_b) > min(h_a, h_b) * 2.0:
        return False
    if a["x1"] < b["x0"]:
        gap = b["x0"] - a["x1"]
    elif b["x1"] < a["x0"]:
        gap = a["x0"] - b["x1"]
    else:
        return True
    return gap <= gap_max


def _rebuild_resolved(blob_ids: list[str], blob_meta: dict[str, dict]) -> list[list[dict]]:
    return [[blob_meta[bid] for bid in blob_ids if bid in blob_meta]]


def apply_rules(order: dict, blobs: list[dict],
                gap_max: int = 48,
                verbose: bool = True) -> tuple[dict, dict]:
    """order JSON にルールを順次適用して新 order を返す。

    Returns:
        (new_order, stats)
    """
    blob_meta = _blob_meta_index(blobs)
    groups = [dict(g) for g in order.get("groups", [])]
    stats = {
        "r1_dup_removed": 0,
        "r2_frame_merged": 0,
        "r3_horizontal_merged": 0,
        "r3_horizontal_skipped_container": 0,
        "r3_horizontal_skipped_anim": 0,
        "r3_horizontal_warn_multi": 0,
        "r4_orphan_added": 0,
        "r5_label_normalized": 0,
    }
    log: list[str] = []

    # 各 group の blob IDs をリスト化 (mutable)
    group_blob_lists: list[list[str]] = [_group_blob_ids(g) for g in groups]

    # === R1: 重複 blob 除去 ===
    seen_blob: dict[str, int] = {}
    for gi, blob_ids in enumerate(group_blob_lists):
        new_ids = []
        for bid in blob_ids:
            if bid in seen_blob:
                stats["r1_dup_removed"] += 1
                log.append(f"  R1 dup: blob {bid} 2回目登場 g{gi} → 除去 (first g{seen_blob[bid]})")
                continue
            seen_blob[bid] = gi
            new_ids.append(bid)
        group_blob_lists[gi] = new_ids

    # === R2: 枠内包 merge (group 単位判定) ===
    # 各 group の代表 blob (最大 area) を取り、 包含関係をチェック
    def _group_representative(blob_ids: list[str]) -> dict | None:
        candidates = [blob_meta[bid] for bid in blob_ids if bid in blob_meta]
        return max(candidates, key=_bbox_area) if candidates else None

    merged_into: dict[int, int] = {}  # gi → 統合先 gi
    for gi, blob_ids_i in enumerate(group_blob_lists):
        rep_i = _group_representative(blob_ids_i)
        if rep_i is None or _bbox_area(rep_i) < 5000:
            continue
        if not _is_container_group(groups[gi], blob_meta):
            continue  # 枠 group でなければ R2 対象外
        for gj, blob_ids_j in enumerate(group_blob_lists):
            if gj == gi or gj in merged_into:
                continue
            rep_j = _group_representative(blob_ids_j)
            if rep_j is None:
                continue
            if _is_container_group(groups[gj], blob_meta):
                continue  # 両方 container なら別構造として skip
            if _bbox_overlap_ratio(rep_i, rep_j) > 0.70:
                merged_into[gj] = gi
                stats["r2_frame_merged"] += 1
                log.append(f"  R2 frame: g{gj}「{groups[gj].get('label')}」 → g{gi}「{groups[gi].get('label')}」 (枠内包 70%+)")

    # merged_into を反映: gj の blob を gi の末尾に追加 (重複防止)
    new_group_blob_lists: list[list[str]] = []
    keep_indices: list[int] = []
    for gi, blob_ids in enumerate(group_blob_lists):
        if gi in merged_into:
            continue
        keep_indices.append(gi)
    # 統合先に blob を移動
    survivor_map: dict[int, list[str]] = {gi: list(group_blob_lists[gi]) for gi in keep_indices}
    for gj, gi_target in merged_into.items():
        for bid in group_blob_lists[gj]:
            if bid not in survivor_map[gi_target]:
                survivor_map[gi_target].append(bid)
    # 順序を保ちつつ生き残り group のみに
    group_blob_lists = [survivor_map[gi] for gi in keep_indices]
    groups = [groups[gi] for gi in keep_indices]

    # === R3: 水平方向近接 merge (group 単位判定) ===
    # blob → group index map (現状態)
    blob_to_group: dict[str, int] = {}
    for gi, ids in enumerate(group_blob_lists):
        for bid in ids:
            blob_to_group[bid] = gi

    # 全 blob ペアで水平近接を判定し、 group ペアごとに「跨ぎ件数」 を集計
    crossing_pairs: dict[tuple[int, int], int] = defaultdict(int)
    blob_crossings: dict[str, set[int]] = defaultdict(set)  # blob が跨る group 集合
    for i, a in enumerate(blobs):
        for b in blobs[i + 1:]:
            if not _horizontal_neighbor(a, b, gap_max, 25):
                continue
            ga = blob_to_group.get(a["id"])
            gb = blob_to_group.get(b["id"])
            if ga is None or gb is None or ga == gb:
                continue
            key = (min(ga, gb), max(ga, gb))
            crossing_pairs[key] += 1
            blob_crossings[a["id"]].add(ga)
            blob_crossings[a["id"]].add(gb)
            blob_crossings[b["id"]].add(ga)
            blob_crossings[b["id"]].add(gb)

    # group ペア単位で merge 判定
    pair_merge: list[tuple[int, int, int]] = []  # (src, dst, count)
    for (g1, g2), cnt in crossing_pairs.items():
        # container guard (verifier 指摘 axis_2: blob_meta も渡して area判定)
        if _is_container_group(groups[g1], blob_meta) or _is_container_group(groups[g2], blob_meta):
            stats["r3_horizontal_skipped_container"] += 1
            log.append(f"  R3 skip-container: g{g1} ⇔ g{g2} ({cnt}組)")
            continue
        # anim 型違い + label prefix違い なら 意図的演出として skip
        anim1 = groups[g1].get("anim", "")
        anim2 = groups[g2].get("anim", "")
        if anim1 != anim2:
            stats["r3_horizontal_skipped_anim"] += 1
            log.append(f"  R3 skip-anim: g{g1}({anim1}) ⇔ g{g2}({anim2}) ({cnt}組)")
            continue
        # 3 group 以上跨ぎ blob があれば WARN+skip
        multi_cross = any(len(gs) >= 3 for gs in blob_crossings.values()
                          if g1 in gs and g2 in gs)
        if multi_cross:
            stats["r3_horizontal_warn_multi"] += 1
            log.append(f"  R3 WARN multi: g{g1} ⇔ g{g2} の blob が 3+ group 跨ぎ → 人間判断要")
            continue
        pair_merge.append((g2, g1, cnt))  # 後ろを前に統合

    # pair_merge を適用 (Union-Find 簡易版)
    parent = list(range(len(groups)))
    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # 小さい index を 親にして 順序保つ
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb
    for src, dst, _ in pair_merge:
        _union(src, dst)
        stats["r3_horizontal_merged"] += 1
        log.append(f"  R3 merge: g{src}「{groups[src].get('label')}」 → g{dst}「{groups[dst].get('label')}」")

    # 統合結果を反映
    cluster_map: dict[int, list[int]] = defaultdict(list)
    for gi in range(len(groups)):
        cluster_map[_find(gi)].append(gi)
    new_groups: list[dict] = []
    new_blob_lists: list[list[str]] = []
    for root_gi in sorted(cluster_map.keys()):
        members = sorted(cluster_map[root_gi])
        # 代表 group の meta を引き継ぐ
        base = dict(groups[root_gi])
        merged_blob_ids: list[str] = []
        for gi in members:
            for bid in group_blob_lists[gi]:
                if bid not in merged_blob_ids:
                    merged_blob_ids.append(bid)
        new_groups.append(base)
        new_blob_lists.append(merged_blob_ids)
    groups = new_groups
    group_blob_lists = new_blob_lists

    # === R4: 取りこぼし blob を 最近接 group に統合 (verifier 指摘 + 人手レビュー反映) ===
    # 旧版は「末尾 fade_in」 だったが、 これは取りこぼし要素を「最後に取って付けた感」 で出す
    # 構造で、 「意味のあるまとまり (動画1本 等)」 を破壊する。
    # 改善版: 取りこぼし blob から各 group の最近接 blob までの最小距離を計算し、
    # 200px 以内に group があれば統合 (= その group と同じタイミング/anim で reveal)、
    # それより遠ければ末尾 fade_in に追加 (旧来挙動)。
    used: set[str] = set()
    for ids in group_blob_lists:
        used.update(ids)
    orphans = [b["id"] for b in blobs if b["id"] not in used and b.get("area", 0) > 1000]
    PROXIMITY_PX = 200
    far_orphans: list[str] = []
    if orphans:
        for orphan_id in orphans:
            orphan = blob_meta.get(orphan_id)
            if orphan is None:
                continue
            ocx = (orphan["x0"] + orphan["x1"]) / 2
            ocy = (orphan["y0"] + orphan["y1"]) / 2
            best_gi = -1
            best_dist = float("inf")
            for gi, ids in enumerate(group_blob_lists):
                # container group には 取りこぼしを 加えない (枠に内部要素を 後付けで足さない)
                if _is_container_group(groups[gi], blob_meta):
                    continue
                for bid in ids:
                    b = blob_meta.get(bid)
                    if b is None:
                        continue
                    # 最近接 = blob 同士の bbox 中心距離
                    bcx = (b["x0"] + b["x1"]) / 2
                    bcy = (b["y0"] + b["y1"]) / 2
                    dist = ((ocx - bcx) ** 2 + (ocy - bcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_gi = gi
            if best_gi >= 0 and best_dist <= PROXIMITY_PX:
                group_blob_lists[best_gi].append(orphan_id)
                stats["r4_orphan_added"] = stats.get("r4_orphan_added", 0) + 1
                log.append(f"  R4 orphan-merge: blob {orphan_id} → g{best_gi}「{groups[best_gi].get('label')}」 (dist={best_dist:.0f}px)")
            else:
                far_orphans.append(orphan_id)
                log.append(f"  R4 orphan-far: blob {orphan_id} 近接 group なし (best dist={best_dist:.0f}px) → 末尾 fade_in 候補")
        if far_orphans:
            groups.append({
                "label": f"merged_orphans_far_{len(far_orphans)}",
                "anim": "fade_in",
                "blob_filter": "manual:" + ",".join(far_orphans),
            })
            group_blob_lists.append(far_orphans)
            stats["r4_orphan_tail_added"] = len(far_orphans)
            log.append(f"  R4 orphan-tail: {len(far_orphans)} blob を末尾 fade_in に追加 ({far_orphans[:8]}...)")

    # === R5: 機械的ラベル正規化 ===
    for gi, g in enumerate(groups):
        label = g.get("label", "")
        if MECHANICAL_LABEL_RE.search(label):
            new_label = f"merged_g{gi:02d}"
            log.append(f"  R5 label: g{gi}「{label}」 → 「{new_label}」")
            g["label"] = new_label
            stats["r5_label_normalized"] += 1

    # === resolved_blobs を再構築 ===
    for gi, g in enumerate(groups):
        ids = group_blob_lists[gi]
        g["blob_filter"] = "manual:" + ",".join(ids)
        g["resolved_blobs"] = _rebuild_resolved(ids, blob_meta)

    new_order = dict(order)
    new_order["groups"] = groups

    if verbose:
        for line in log:
            print(line)

    return new_order, stats


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


def _slide_ids(topic_dir: Path, only: list[str] | None) -> list[str]:
    orders_dir = topic_dir / "work" / "orders"
    ids = sorted({
        p.stem.replace("_order", "").replace("slide_", "")
        for p in orders_dir.glob("slide_*_order.json")
    })
    if only:
        want = {f"{int(s):02d}" for s in only}
        ids = [i for i in ids if i in want]
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--slides", default=None)
    ap.add_argument("--gap", type=int, default=48,
                    help="水平 merge gap 上限 (px). default 48. --aggressive で 80 に上書き")
    ap.add_argument("--aggressive", action="store_true",
                    help="gap を 80px に上書き (横3カラム見出しの誤merge リスク高)")
    ap.add_argument("--dry-run", action="store_true",
                    help="order JSON 上書きせず stats だけ出力")
    args = ap.parse_args(argv)

    gap_max = 80 if args.aggressive else args.gap
    topic_dir = _topic_dir(args.topic)
    only = [s.strip() for s in args.slides.split(",")] if args.slides else None
    ids = _slide_ids(topic_dir, only)
    if not ids:
        sys.stderr.write(f"対象 slide 不在 (topic={args.topic})\n")
        return 1

    print(f"=== order_merge ({args.topic}, gap={gap_max}, dry={args.dry_run}, {len(ids)} slides) ===")
    agg = defaultdict(int)
    for sid in ids:
        orders_dir = topic_dir / "work" / "orders"
        order_path = orders_dir / f"slide_{sid}_order.json"
        blobs_path = topic_dir / "work" / "blobs" / f"slide_{sid}" / "blobs.json"
        if not order_path.exists() or not blobs_path.exists():
            print(f"  [skip] slide_{sid}: order or blobs 不在")
            continue
        order = json.loads(order_path.read_text())
        blobs = json.loads(blobs_path.read_text())["blobs"]
        print(f"\n[slide_{sid}] groups: {len(order.get('groups', []))}")
        new_order, stats = apply_rules(order, blobs, gap_max=gap_max, verbose=True)
        print(f"  → groups: {len(new_order['groups'])}, stats: {dict(stats)}")
        for k, v in stats.items():
            agg[k] += v
        if not args.dry_run:
            bak = order_path.with_suffix(".json.bak")
            if not bak.exists():
                bak.write_text(order_path.read_text())
            order_path.write_text(json.dumps(new_order, ensure_ascii=False, indent=2))

    print(f"\n=== 集計 ({len(ids)} slides) ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
