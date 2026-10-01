"""台本(lines/title) の変更を検出して slide_prompt を claude CLI で 再生成する Phase 0。

build_all chain の 最先頭で 実行。 ユーザーが UI で セリフを編集 → slide_prompt が
古いまま → 画像と台本が矛盾、 という典型事故を防ぐ。

検出ロジック:
  hash = sha256(title + "|" + "\\n".join(captions))
  各 post 個別 slide で:
    stored = slide.get("slide_prompt_lines_hash")
    if stored is None:                # backfill (初回)
      slide["slide_prompt_lines_hash"] = current  → 再生成しない
    elif stored != current and slide.get("slide_prompt"):
      claude CLI で slide_prompt を最新セリフに合わせて 編集 → hash 更新

shared_from がある slide は skip(共通プールは別管理)。

Usage:
  python -m src.post_video.sync_slide_prompts --topic <post-id> [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import paths
from src.claude_cli import run_llm_cli

DEFAULT_MODEL = "claude-sonnet-4-5"
TIMEOUT_PER_SLIDE = 180  # 3 分


def _lines_hash(slide: dict) -> str:
    title = (slide.get("title") or "").strip()
    captions = [(l.get("caption") or "").strip() for l in slide.get("lines", [])]
    payload = title + "|" + "\n".join(captions)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _regenerate_one(slide: dict, timeout: int = TIMEOUT_PER_SLIDE) -> tuple[bool, str]:
    """claude CLI で 1 slide の slide_prompt を更新。 Returns (success, new_prompt or error)."""
    cur_prompt = (slide.get("slide_prompt") or "").strip()
    cur_title = (slide.get("title") or "").strip()
    captions_block = "\n".join(f"  - {l.get('caption', '')}" for l in slide.get("lines", []))
    user_prompt = (
        "次の動画スライドの 「画像 指示文 (slide_prompt)」 を、 最新のセリフ内容に合わせて 更新してください。\n\n"
        f"[スライドタイトル]\n{cur_title}\n\n"
        f"[最新の字幕 (caption) 群]\n{captions_block}\n\n"
        f"[現在の画像指示文 (古い、 セリフ編集前の もの)]\n{cur_prompt}\n\n"
        "[指示]\n"
        "現在の画像指示文を ベースに、 最新の字幕内容と矛盾しないように 修正してください。 "
        "ベースの絵柄・構図・色味は 可能な限り 維持し、 字幕で 新しく出てきた要素・数字・キーワードを 反映。 "
        "字幕で 言及されなくなった 要素は 削除 or 控えめに。\n\n"
        "[出力形式]\n"
        "JSON のみ。 説明文 一切なし。 ```json``` 等の ブロックも 不要。 "
        '形式: {"slide_prompt": "..."}\n'
        "1 段落の自然言語、 gpt-image-2 に そのまま投げる前提。 日本語混在 OK。"
    )
    try:
        raw = run_llm_cli(
            "", user_prompt, model=DEFAULT_MODEL, timeout=timeout,
        )
    except RuntimeError as exc:
        return False, f"LLM failed: {exc}"

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    try:
        update = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"JSON parse 失敗: {e} / raw[:200]: {raw[:200]}"
    new_prompt = (update.get("slide_prompt") or "").strip()
    if not new_prompt:
        return False, "slide_prompt 空"
    return True, new_prompt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="検出のみ、 書き戻ししない")
    args = ap.parse_args(argv)

    topic = args.topic
    slides_path = paths.slides_json(topic)
    if not slides_path.exists():
        print(f"ERROR: slides.json not found: {slides_path}", file=sys.stderr)
        return 1

    data = json.loads(slides_path.read_text(encoding="utf-8"))
    slides = data.get("slides", [])

    backfill: list[int] = []  # hash 不在 → 初回 backfill (再生成しない)
    stale: list[int] = []     # hash mismatch → 再生成対象

    for idx, s in enumerate(slides):
        if not isinstance(s, dict):
            continue  # section marker (e.g. "[SECTION_INTRO]") は skip
        if s.get("shared_from"):
            continue  # shared プールは別管理
        if not s.get("slide_prompt"):
            continue  # そもそも prompt が無い slide は対象外 (後で post-scripter が埋める)
        cur = _lines_hash(s)
        stored = s.get("slide_prompt_lines_hash")
        if stored is None:
            backfill.append(idx)
            s["slide_prompt_lines_hash"] = cur  # 初回 backfill
        elif stored != cur:
            stale.append(idx)

    print(f"=== sync_slide_prompts (topic={topic}) ===")
    print(f"  backfill (hash 不在 → 初回登録 / 再生成しない): {len(backfill)} 件")
    print(f"  stale (hash 不一致 → 再生成対象): {len(stale)} 件")

    if backfill and not args.dry_run:
        # backfill だけは hash を保存
        slides_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  [write] backfill {len(backfill)} 件 を slides.json に保存")

    if not stale:
        print("→ 再生成対象なし、 done")
        return 0

    if args.dry_run:
        for idx in stale:
            s = slides[idx]
            print(f"  [dry-run] sid={s.get('sid')} title={s.get('title','')[:30]}")
        return 0

    print(f"→ {len(stale)} 件を claude CLI 並列 {args.workers} で再生成")
    results: dict[int, tuple[bool, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_regenerate_one, slides[idx]): idx for idx in stale}
        for fut in as_completed(futs):
            idx = futs[fut]
            sid = slides[idx].get("sid", f"raw_{idx}")
            try:
                ok, payload = fut.result()
            except Exception as e:
                ok, payload = False, f"EXCEPTION: {type(e).__name__}: {e}"
            results[idx] = (ok, payload)
            tag = "OK" if ok else "FAIL"
            print(f"  [{tag}] sid={sid}: {payload[:80] if ok else payload}")

    # 成功分を slides.json に書き戻し + hash 更新
    n_ok = 0
    for idx, (ok, payload) in results.items():
        if not ok:
            continue
        slides[idx]["slide_prompt"] = payload
        slides[idx]["slide_prompt_lines_hash"] = _lines_hash(slides[idx])
        n_ok += 1
    if n_ok:
        slides_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_fail = len(stale) - n_ok
    print(f"\n=== done: {n_ok} 件 更新 / {n_fail} 件 失敗 ===")
    # 失敗があっても build_all chain は止めない (古い slide_prompt のままで image regen される)
    return 0


if __name__ == "__main__":
    sys.exit(main())
