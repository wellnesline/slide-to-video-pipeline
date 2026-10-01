"""blob ID 番号 overlay PNG 生成 — subagent prompt の 2枚目用。

verifier  採択 案β。 改善点:
  - ID は bbox 外側にラベル + リーダー線 (本体ピクセル隠さない)
  - 白縁取り+赤地+白文字 (3色強コントラスト)
  - bbox 自体は薄い色のアウトラインで囲う

Usage:
  python tools/anim_quality/generate_overlay.py --topic <post-id> --slides 05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def _font(size: int = 22) -> ImageFont.FreeTypeFont:
    # フォントは環境変数 OVERLAY_FONT_PATH で渡す(既定値なし。未設定なら PIL の既定フォント)
    candidates = [os.environ.get("OVERLAY_FONT_PATH", "")]
    for c in candidates:
        if c and Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _place_label(b: dict, W: int, H: int, label_w: int, label_h: int,
                 occupied: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """ID ラベルの配置位置を決める。 bbox 外側 4方向 (上下左右) で空いてる場所を選ぶ。
    occupied = 既に置かれたラベル box list (重なり回避用)。
    """
    x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
    margin = 4
    candidates = [
        # (lx, ly) ラベル左上座標
        (x0, max(0, y0 - label_h - margin)),       # 上
        (max(0, x0 - label_w - margin), y0),       # 左
        (min(W - label_w, x1 + margin), y0),       # 右
        (x0, min(H - label_h, y1 + margin)),       # 下
    ]
    for lx, ly in candidates:
        lx_e, ly_e = lx + label_w, ly + label_h
        if lx < 0 or ly < 0 or lx_e > W or ly_e > H:
            continue
        overlap = False
        for ox0, oy0, ox1, oy1 in occupied:
            if lx < ox1 and lx_e > ox0 and ly < oy1 and ly_e > oy0:
                overlap = True
                break
        if not overlap:
            return (lx, ly, lx_e, ly_e)
    # 全部失敗 → bbox 内左上 (本体上に被せる、 最後の手段)
    lx = x0 + 2
    ly = y0 + 2
    return (lx, ly, lx + label_w, ly + label_h)


def make_overlay(raw_path: Path, blobs: list[dict], out_path: Path) -> None:
    raw = Image.open(raw_path).convert("RGB")
    W, H = raw.size
    # raw を 70% の明度で fade させて、 overlay を載せやすく
    arr = np.array(raw, dtype=np.float32)
    fade = (arr * 0.55 + np.full_like(arr, 255) * 0.45).astype(np.uint8)
    img = Image.fromarray(fade).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(20)

    # 描画順: large area → small area の順 (小 blob のラベル が大 blob のラベルに上書きされても OK)
    sorted_blobs = sorted(blobs, key=lambda b: -((b["x1"]-b["x0"]) * (b["y1"]-b["y0"])))

    # bbox アウトライン (薄いシアン)
    for b in sorted_blobs:
        draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]],
                       outline=(0, 130, 200), width=2)

    # ラベル配置
    occupied: list[tuple[int, int, int, int]] = []
    label_w, label_h = 36, 22
    for b in sorted_blobs:
        text = b["id"]
        lx, ly, lx_e, ly_e = _place_label(b, W, H, label_w, label_h, occupied)
        # リーダー線 (ラベルから bbox 中央へ)
        bcx = (b["x0"] + b["x1"]) // 2
        bcy = (b["y0"] + b["y1"]) // 2
        label_cx = (lx + lx_e) // 2
        label_cy = (ly + ly_e) // 2
        draw.line([(label_cx, label_cy), (bcx, bcy)], fill=(160, 160, 160), width=1)
        # ラベル箱 (白縁取り + 赤地 + 白文字)
        draw.rectangle([lx-2, ly-2, lx_e+2, ly_e+2], fill=(255, 255, 255))
        draw.rectangle([lx, ly, lx_e, ly_e], fill=(220, 30, 30))
        # 文字
        draw.text((lx + 4, ly + 1), text, fill=(255, 255, 255), font=font)
        occupied.append((lx, ly, lx_e, ly_e))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


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
    args = ap.parse_args(argv)

    topic_dir = _topic_dir(args.topic)
    blobs_root = topic_dir / "work" / "blobs"
    slides_raw_dir = topic_dir / "slides_raw"
    overlay_dir = topic_dir / "work" / "blobs_overlay"
    only = [f"{int(s):02d}" for s in args.slides.split(",")] if args.slides else None

    targets = sorted(blobs_root.glob("slide_*/blobs.json"))
    print(f"=== generate_overlay ({args.topic}, {len(targets)} slides) ===")
    n = 0
    for f in targets:
        sid = f.parent.name.replace("slide_", "")
        if only and sid not in only:
            continue
        raw = slides_raw_dir / f"{sid}.png"
        if not raw.exists():
            print(f"  [skip] slide_{sid}: raw 不在")
            continue
        blobs = json.loads(f.read_text())["blobs"]
        out = overlay_dir / f"slide_{sid}_overlay.png"
        make_overlay(raw, blobs, out)
        try:
            rel = out.relative_to(ROOT)
        except ValueError:
            rel = out
        print(f"  [ok] slide_{sid} → {rel} ({len(blobs)} blobs)")
        n += 1
    print(f"=== done: {n} slides ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
