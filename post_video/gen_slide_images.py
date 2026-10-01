"""<post-id> の post 個別 slide raw png を gpt-image-2 で生成。

slides.json の各 post 個別 slide(shared_from なし)の `slide_prompt` を
OpenAI gpt-image-2 に投げて 1536×1024 png を `slides_raw/<NN>.png` に保存。

shared slide(shared_from)はスキップ(shared プールに既に raw.png 完備)。

既に存在する png は `--force` がなければスキップ(再生成しない)。

★ hardlock: --quality CLI 引数撤去、DEFAULT_IMAGE_QUALITY (low) 固定。
   過去にマスターガイドが非 low 品質を例示していて約4倍コスト膨張
   が発生したため、AI/人間が非 low 品質を渡せないようにした。変更は change-validator 経由のみ。

Usage:
  python -m src.post_video.gen_slide_images --topic <post-id>
  python -m src.post_video.gen_slide_images --topic <post-id> --force
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import paths
from .slides import load_slides


def _prompt_sha(prompt: str, common_style: str) -> str:
    """slide_prompt + COMMON_STYLE の sha (cache invalidation 用)。"""
    h = hashlib.sha256((prompt + "||" + common_style).encode("utf-8"))
    return h.hexdigest()[:16]


def _png_meta_path(out: Path) -> Path:
    return out.with_suffix(".meta.json")


def _png_cache_valid(out: Path, prompt: str, common_style: str) -> bool:
    """png 既存 + meta の prompt_sha が一致するか。"""
    if not out.exists():
        return False
    meta_p = _png_meta_path(out)
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return False
    return meta.get("prompt_sha") == _prompt_sha(prompt, common_style)


def _write_png_meta(out: Path, prompt: str, common_style: str) -> None:
    _png_meta_path(out).write_text(json.dumps({
        "prompt_sha": _prompt_sha(prompt, common_style),
    }, ensure_ascii=False, indent=2))


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


def force_delete_sid_image(sid: str, slides: list[dict], out_dir: Path) -> int:
    """sid 個別 slide の png + meta を削除。

    削除件数を返す。 shared セクションは保護(削除しない)。
    """
    target = next((s for s in slides if s["id"] == sid), None)
    if target is None:
        sys.stderr.write(f"[--force-slides] WARN: sid {sid} は slides.json にありません 無視\n")
        return 0
    if target["shared_id"]:
        print(f"  [skip force] sid={sid}: shared セクション (shared_from), プールは保護")
        return 0
    n = 0
    for p in (out_dir / f"{sid}.png", _png_meta_path(out_dir / f"{sid}.png")):
        if p.exists():
            p.unlink()
            n += 1
    return n

# .env ロードは共通ローダー(src/_dotenv.py)に集約。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dotenv import load_dotenv as _load_dotenv  # noqa: E402
from openai_image_config import DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_QUALITY  # noqa: E402

# スタイル句は中立なサンプル。実際の配色は環境変数 POST_VIDEO_STYLE_PALETTE で渡す。
_STYLE_PALETTE = os.environ.get("POST_VIDEO_STYLE_PALETTE", "チャコール・ティール・ライトグレーのうち2-3色のみ使用")

COMMON_STYLE = f"""
スタイル: 手書き風のシンプルな図解。
- 線が少しガタついた手描きストローク
- 箱は白背景に細い枠線、淡い塗り
- カラー: {_STYLE_PALETTE}
- 日本語フォントは太め、大きめ
- 装飾絵文字なし、3Dなし、クリップアートなし
レイアウト指示:
- 横長キャンバス。要素は中央 60% 領域に集約(上下 20% は余白)
- 文字大きめ、スマホでも読めるサイズ
- 余白多め
"""


def _gen_one(client, sid: str, prompt: str, out: Path) -> str:
    """1 枚生成。 thread worker から呼ばれる。quality は DEFAULT_IMAGE_QUALITY (low) 固定。"""
    full_prompt = f"{prompt}\n\n{COMMON_STYLE}"
    t0 = time.time()
    try:
        r = client.images.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=full_prompt,
            size="1536x1024",
            quality=DEFAULT_IMAGE_QUALITY,
            n=1,
        )
        out.write_bytes(base64.b64decode(r.data[0].b64_json))
        # content_hash meta 書き出し(次回 cache 判定用、 prompt 変更検出)
        _write_png_meta(out, prompt, COMMON_STYLE)
        return f"  [ok] {sid} ({time.time() - t0:.1f}s, {out.stat().st_size // 1024}KB)"
    except Exception as e:
        return f"  [error] {sid}: {e}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--force", action="store_true",
                    help="既存 png を全 sid 強制再生成 (--force-slides より優先)")
    ap.add_argument("--force-slides", default=None,
                    help="カンマ区切り sid。 指定 slide の png + meta を事前削除してから再生成。 "
                         "例: --force-slides 03,07 (shared セクションは保護、 --force と併用なら --force 優先)")
    ap.add_argument("--only-slides", default=None,
                    help="カンマ区切り sid。 指定 slide のみ処理、 他 sid は完全 skip。 "
                         "UI の per-scene 再生成ボタン用、 --force-slides と併用で アトミック再生成")
    ap.add_argument("--workers", type=int, default=4,
                    help="並列 API 呼び出し数(default 4、OpenAI rate limit 内推奨)")
    args = ap.parse_args(argv)

    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.stderr.write("OPENAI_API_KEY が未設定(.env を確認)\n")
        return 1

    out_dir = paths.slides_raw_dir(args.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = load_slides(args.topic)
    targets = [s for s in slides if s["shared_id"] is None]

    # === --force-slides: 指定 slide の png + meta を事前削除(post 個別のみ)===
    force_slides = _parse_force_slides(args.force_slides)
    if args.force and force_slides:
        sys.stderr.write("[--force-slides] WARN: --force と併用、 --force 優先(全 sid 強制再生成)\n")
        force_slides = None
    if force_slides:
        print(f"[--force-slides] {sorted(force_slides)} の png/meta を削除")
        deleted_total = 0
        for sid in sorted(force_slides):
            deleted_total += force_delete_sid_image(sid, slides, out_dir)
        print(f"  → {deleted_total} 件削除")

    # === --only-slides: 指定 sid 以外を完全 skip ===
    only_slides = _parse_force_slides(args.only_slides)
    if only_slides:
        before = len(targets)
        targets = [s for s in targets if s["id"] in only_slides]
        print(f"[--only-slides] {sorted(only_slides)} のみ処理 ({before} → {len(targets)} slides)")
        if not targets:
            print("  (該当 slide 無し、 何もせず終了)")
            return 0

    # filter: 既存 png + content_hash 一致なら skip(sid 並び替え事故対策)
    # 旧来は out.exists() だけで skip していたが、 slide_prompt 書き換え時に旧 png が
    # 新 slide に紐付く致命事故由来。 meta.json の prompt_sha と一致しない時のみ再生成。
    pending: list[tuple[str, str, Path]] = []
    n_skip = 0
    for s in targets:
        sid = s["id"]
        out = out_dir / f"{sid}.png"
        prompt = (s.get("slide_prompt") or s.get("title") or "").strip()
        if not args.force and _png_cache_valid(out, prompt, COMMON_STYLE):
            n_skip += 1
            continue
        pending.append((sid, prompt, out))

    print(f"=== gen_slide_images (topic={args.topic}, quality={DEFAULT_IMAGE_QUALITY} [fixed], workers={args.workers}) ===")
    print(f"  total: {len(slides)} slides, post 個別: {len(targets)} slides")
    print(f"  skip (cached): {n_skip}, pending: {len(pending)}")
    print()

    if not pending:
        print("=== done(全 cache hit)===")
        return 0

    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = OpenAI()
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_gen_one, client, sid, prompt, out)
                   for sid, prompt, out in pending]
        for f in as_completed(futures):
            msg = f.result()
            print(msg, flush=True)
            if msg.startswith("  [error]"):
                n_fail += 1

    print(f"\n=== done: {len(pending) - n_fail} ok / {n_fail} fail ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
