"""<post-id> の全 slide blob anim を 6並列でレンダリング。

shared slide の anim は shared プールから直接 copy(再 render しない)。
post 個別 slide のみ blob_renderer で生成。

cache 戦略(監査 1 由来、 sid_li bind bug 対策):
  - anim.meta.json に {order_sha, raw_sha, labels_sha, duration} を焼く
  - 次回 render 時、 3 sha + duration が一致したら skip、 不一致なら再 render
  - 旧来の mtime 比較は line 並び替え時に旧 anim 残存する致命事故由来で撤廃

blob サイレント skip ガード(v2 事故根治):
  - post 個別 slide の skip 件数が閾値を超えると exit 1 BLOCK
  - 閾値: BLOB_SKIP_BLOCK_RATIO(default 0.5)×  post_individual_count かつ
    BLOB_SKIP_BLOCK_MIN(default 3) 以上 skip で発動
  - 環境変数 BLOB_SKIP_BLOCK_RATIO / BLOB_SKIP_BLOCK_MIN で閾値上書き可
  - --skip-blob-check は 物理削除(5回の静止画フォールバック事故根治)
    緊急 escape は別スクリプトへ隔離する設計、 本flag は復活させない

Usage:
  python -m post_video.parallel_render --topic <post-id> [--workers 6]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Any

from . import paths
from .slides import load_slides


def _file_sha(p: Path) -> str:
    """ファイルの sha256 (16 文字 prefix)。 cache invalidation 用。"""
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _anim_meta_path(anim_out: Path) -> Path:
    return anim_out.with_suffix(".meta.json")


def _read_anim_meta(anim_out: Path) -> dict | None:
    p = _anim_meta_path(anim_out)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_anim_meta(anim_out: Path, meta: dict) -> None:
    _anim_meta_path(anim_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _parse_force_slides(arg: str | None) -> set[str] | None:
    """'03,07,12' → {'03','07','12'}。 単桁 '3' も '03' に正規化。 None / '' → None。"""
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
            sys.stderr.write(f"[--force-slides] WARN: invalid sid {t!r} 無視\n")
    return out or None


def force_delete_sid_anim(sid: str, slides: list[dict], topic: str) -> int:
    """sid 個別 slide の anim mp4 + meta を削除。

    削除件数を返す。 shared セクションは保護(削除しない)。
    blob_dir / orders json は削除しない(削ると deps missing で render 不能になるため)。
    """
    target = next((s for s in slides if s["id"] == sid), None)
    if target is None:
        sys.stderr.write(f"[--force-slides] WARN: sid {sid} は slides.json にありません 無視\n")
        return 0
    if target["shared_id"]:
        print(f"  [skip force] sid={sid}: shared セクション (shared_from), プールは保護")
        return 0
    n = 0
    anim = anim_path_for_sid(topic, sid)
    for p in (anim, _anim_meta_path(anim)):
        if p.exists():
            p.unlink()
            n += 1
    return n


def anim_path_for_sid(topic: str, sid: str) -> Path:
    """sid 別 anim mp4 の path(test からも参照)。"""
    from . import paths as _paths_mod  # 循環回避
    return _paths_mod.anim_dir(topic) / f"{sid}.mp4"


def _anim_cache_valid(anim_out: Path, raw: Path, labels: Path, order_path: Path, duration: float) -> bool:
    """anim 既存 + meta の sha 3 つ + duration が現入力と一致するか。"""
    if not anim_out.exists():
        return False
    meta = _read_anim_meta(anim_out)
    if not meta:
        return False
    return (
        meta.get("order_sha") == _file_sha(order_path)
        and meta.get("raw_sha") == _file_sha(raw)
        and meta.get("labels_sha") == _file_sha(labels)
        and abs(meta.get("duration", 0) - duration) < 0.05
    )

ROOT = paths.repo_root()

HEAD_PAD = 0.05
SRC_DIR = ROOT / "src"

# blob サイレント skip BLOCK ガード閾値(環境変数で上書き可)
import os as _os
BLOB_SKIP_BLOCK_RATIO: float = float(_os.environ.get("BLOB_SKIP_BLOCK_RATIO", "0.5"))
BLOB_SKIP_BLOCK_MIN: int = int(_os.environ.get("BLOB_SKIP_BLOCK_MIN", "3"))


def _slide_duration(audio_paths: list[Path], lines: list[dict]) -> float:
    cur = 0.0
    for li, line in enumerate(lines):
        wav = audio_paths[li]
        if not wav.exists():
            continue
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", str(wav)],
            capture_output=True, text=True,
        )
        d = float(json.loads(r.stdout)["format"]["duration"]) + HEAD_PAD
        cap = (line.get("caption") or "").strip()
        last = cap[-1] if cap else ""
        if last in "。!?!?":
            gap = 0.40
        elif last in "、・":
            gap = 0.20
        else:
            gap = 0.15
        cur = cur + d + gap
    return cur + 0.7


def _render_one(args: tuple) -> str:
    sid, raw, labels, order_path, anim_out, duration, shared_anim_src = args

    # === shared slide ===
    if shared_anim_src is not None:
        if shared_anim_src.exists():
            # shared anim 既存 → topic anim_out へ copy (content_hash 比較で再 copy 判定)
            # 旧来は mtime 比較だったが、shutil.copy2 が mtime を保持するため一度 copy 後に
            # shared 側が再 render されても「同じ mtime」 になり再 copy が trigger されない
            # 構造的バグ(outro 3秒問題 由来)。 content_hash で根治。
            need_copy = not anim_out.exists()
            if not need_copy:
                src_sha = _file_sha(shared_anim_src)
                dst_sha = _file_sha(anim_out)
                need_copy = (src_sha != dst_sha)
            if need_copy:
                anim_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shared_anim_src, anim_out)
                # shared 経路でも meta を書き、 source_sha で監査可能に
                _write_anim_meta(anim_out, {
                    "source": "shared",
                    "source_path": str(shared_anim_src),
                    "source_sha": _file_sha(shared_anim_src),
                    "duration": round(duration, 3),
                })
                return f"shared (refreshed) {sid} ({duration:.1f}s)"
            return f"shared (cached) {sid} ({duration:.1f}s)"
        else:
            # shared anim 不在 → fresh render → shared に書き戻して次回再利用 + topic にも copy
            if not (raw.exists() and labels.exists() and order_path.exists()):
                return f"skip {sid}: shared deps missing"
            sys.path.insert(0, str(SRC_DIR))
            from diagram_blob.blob_renderer import render_diagram_anim
            order = json.loads(order_path.read_text())
            target = min(3.0, duration * 0.6)
            try:
                # shared 側に直接 render
                shared_anim_src.parent.mkdir(parents=True, exist_ok=True)
                render_diagram_anim(
                    base_img=raw, labels_npz=labels, order=order,
                    duration=duration, out_mp4=shared_anim_src,
                    auto_stagger=True, stagger_gap=0.7,
                    target_finish_time=target,
                )
                # topic anim_out にも copy
                anim_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shared_anim_src, anim_out)
                return f"shared (fresh render) {sid} ({duration:.1f}s, target={target:.1f}s)"
            except Exception as e:
                return f"FAIL shared {sid}: {type(e).__name__}: {e}"

    # === post 個別 ===
    if not (raw.exists() and labels.exists() and order_path.exists()):
        return f"skip {sid}: deps missing (raw={raw.exists()} labels={labels.exists()} order={order_path.exists()})"

    # content_hash ベース cache 判定(監査 1 由来、 旧 mtime 判定撤廃)
    if _anim_cache_valid(anim_out, raw, labels, order_path, duration):
        return f"cached {sid} (content_hash hit)"

    sys.path.insert(0, str(SRC_DIR))
    from diagram_blob.blob_renderer import render_diagram_anim

    order = json.loads(order_path.read_text())
    anim_out.parent.mkdir(parents=True, exist_ok=True)
    target = min(3.0, duration * 0.6)
    try:
        render_diagram_anim(
            base_img=raw, labels_npz=labels, order=order,
            duration=duration, out_mp4=anim_out,
            auto_stagger=True, stagger_gap=0.7,
            target_finish_time=target,
        )
        # meta 書き出し(次回 cache 判定用)
        _write_anim_meta(anim_out, {
            "order_sha": _file_sha(order_path),
            "raw_sha": _file_sha(raw),
            "labels_sha": _file_sha(labels),
            "duration": round(duration, 3),
        })
        return f"OK {sid} ({duration:.1f}s, target={target:.1f}s)"
    except Exception as e:
        return f"FAIL {sid}: {type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force-slides", default=None,
                    help="カンマ区切り sid。 指定 slide の anim.mp4 + .meta.json を事前削除してから render。 "
                         "例: --force-slides 03,07 (shared セクションは保護、 blob_dir / orders は保護=削除しない)")
    ap.add_argument("--only-slides", default=None,
                    help="カンマ区切り sid。 指定 slide のみ処理、 他 sid は完全 skip (cache 判定もしない)。 "
                         "UI の per-scene 再生成ボタン用。 --force-slides と併用すると 「指定 sid だけアトミック再生成」")
    # --skip-blob-check 物理削除、 BLOCK ガード は常時ON
    args = ap.parse_args(argv)

    topic = args.topic
    slides = load_slides(topic)

    # === --force-slides: 指定 slide の anim を事前削除(post 個別のみ、 shared は保護)===
    force_slides = _parse_force_slides(args.force_slides)
    if force_slides:
        print(f"[--force-slides] {sorted(force_slides)} の anim を削除")
        deleted_total = 0
        for sid in sorted(force_slides):
            deleted_total += force_delete_sid_anim(sid, slides, topic)
        print(f"  → {deleted_total} 件削除")

    # === --only-slides: 指定 slide 以外を完全 skip ===
    only_slides = _parse_force_slides(args.only_slides)
    if only_slides:
        before = len(slides)
        slides = [s for s in slides if s["id"] in only_slides]
        print(f"[--only-slides] {sorted(only_slides)} のみ処理 ({before} → {len(slides)} slides)")
        if not slides:
            print("  (該当 slide 無し、 何もせず終了)")
            return 0

    work_args: list[tuple] = []
    post_individual_count = 0  # shared 以外の post 個別 slide 数
    for s in slides:
        sid = s["id"]
        duration = _slide_duration(s["audio_paths"], s["lines"])
        anim_out = paths.anim_dir(topic) / f"{sid}.mp4"
        shared_anim = s["anim_path"] if s["shared_id"] else None
        if shared_anim is None:
            post_individual_count += 1
        work_args.append((
            sid, s["raw_path"], s["labels_path"], s["order_path"],
            anim_out, duration, shared_anim,
        ))

    print(f"=== parallel render: {len(work_args)} slides × {args.workers} workers "
          f"(topic={topic}, post_individual={post_individual_count}) ===")
    set_start_method("spawn", force=True)
    results: list[str] = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_render_one, work_args):
            print(r, flush=True)
            results.append(r)

    # 結果 summary 集計
    ok_count = sum(1 for r in results if r.startswith(("OK ", "cached ", "shared ")))
    skip_count = sum(1 for r in results if r.startswith("skip "))
    fail_count = sum(1 for r in results if r.startswith("FAIL "))

    # 案 D: resolved_blobs status を summary に追加 (blob_finalize_guard spike)
    # ok になった slide の order JSON を走査して resolved/unresolved 件数を透明化する。
    # 主に 直接起動時の診断用。 run_all 経由では Step 1.85 が先に発火するため
    # ここに到達する時点で unresolved=0 が保証されている(正常ルート)。
    _orders_base = paths.orders_dir(topic)
    _resolved_count = 0
    _unresolved_count = 0
    for _r in results:
        if not _r.startswith(("OK ", "cached ")):
            continue  # shared / skip / FAIL は対象外
        # "OK 05 (3.2s, ...)" or "cached 05 ..." → sid = token[1]
        _tokens = _r.split()
        if len(_tokens) < 2:
            continue
        _sid = _tokens[1]
        _of = _orders_base / f"slide_{_sid}_order.json"
        if not _of.exists():
            continue
        try:
            _od = json.loads(_of.read_text(encoding="utf-8"))
            _has_unresolved = any(
                g.get("blob_filter") and not g.get("resolved_blobs")
                for g in _od.get("groups", [])
            )
            if _has_unresolved:
                _unresolved_count += 1
            else:
                _resolved_count += 1
        except Exception:
            pass

    _resolved_note = ""
    if _resolved_count + _unresolved_count > 0:
        _resolved_note = f" (resolved={_resolved_count} / unresolved={_unresolved_count})"
    print(f"\n=== summary: ok={ok_count}{_resolved_note} / skip(deps missing)={skip_count} / fail={fail_count} ===")

    # unresolved が残っている場合は WARN 表示(exit 2 = WARN, pipeline は継続)
    # run_all 経由では Step 1.85 が finalize 済みのため通常 unresolved=0 のはず
    if _unresolved_count > 0:
        sys.stderr.write(
            f"\n[parallel_render] WARN: {_unresolved_count} 件の slide で resolved_blobs が欠落しています。\n"
            f"  reveal アニメが機能していない可能性があります。\n"
            f"  blob_anim_finalize を実行してから parallel_render を再実行してください:\n"
            f"    python -m src.post_video.blob_anim_finalize --topic {topic}\n"
            f"  run_all 経由で実行した場合は Step 1.85 が自動対処するため、\n"
            f"  この WARN が出た場合は parallel_render を直接起動した可能性があります。\n"
        )
        # WARN のみ(exit 2)、 BLOCK しない(blob deps は揃っているため render 自体は完了している)
        return 2

    # blob サイレント skip BLOCK ガード(v2 事故根治)
    # --skip-blob-check 物理削除、 本ガードは常時ON。 緊急時は人間にエスカレ。
    if post_individual_count >= BLOB_SKIP_BLOCK_MIN:
        skip_ratio = skip_count / post_individual_count if post_individual_count > 0 else 0.0
        if skip_ratio > BLOB_SKIP_BLOCK_RATIO:
            sys.stderr.write(
                f"\n[parallel_render] BLOCK: post 個別 slide の skip 比率が高すぎます\n"
                f"  post_individual={post_individual_count}, skip={skip_count}, "
                f"ratio={skip_ratio:.1%} > threshold={BLOB_SKIP_BLOCK_RATIO:.1%}\n"
                f"  blob deps (labels.npz / order JSON) が揃っていない可能性があります。\n"
                f"  以下の手順で deps を生成してから再実行してください:\n"
                f"    1. python -m src.post_video.blob_anim_prepare --topic {topic}\n"
                f"    2. orderer agent ×N 並列 spawn (orders/*.json 生成)\n"
                f"    3. python -m src.post_video.blob_anim_finalize --topic {topic}\n"
                f"    4. python -m src.post_video.parallel_render --topic {topic}\n"
                f"  ※ --skip-blob-check は 物理削除(5回事故根治)、 緊急時は人間にエスカレ。\n"
                f"  閾値変更: BLOB_SKIP_BLOCK_RATIO(現 {BLOB_SKIP_BLOCK_RATIO}) / "
                f"BLOB_SKIP_BLOCK_MIN(現 {BLOB_SKIP_BLOCK_MIN}) 環境変数で上書き可。\n"
            )
            return 1

    print("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
