"""orderer agent を 並列で 自動 spawn する orchestrator。

入力: <topic>/work/orders/_agent_prompts.md (blob_anim_prepare の出力)
- ``` ... ``` ブロックで囲まれた batch プロンプト群が並んでいる
- 各 batch を 並列で選択中の LLM CLI へ送信（失敗時は反対側へ1回退避）
- 出力を parse_batch_output で order JSON Write

使い方:
    python -m src.post_video.orderer_run --topic <post-id> [--workers 3]

出力: <topic>/work/orders/slide_NN_order.json (N batch x 5 slide まで)
失敗 slide は STDOUT に列挙 (orchestrator が fallback で補完)。
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# blob_anim_prepare の parse 関数を 再利用 (private 名だが orchestrator として呼ぶ)
from src.post_video.blob_anim_prepare import _parse_and_write_batch_output as parse_batch_output  # noqa: E402
from src.post_video import paths  # noqa: E402
from src.claude_cli import run_llm_cli  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-4-5"
TIMEOUT_PER_BATCH = 600  # 10 分(5 slides batch でも余裕を持つ。: 300s で 5-slide batch timeout 事故)


def extract_batches(prompts_md: Path) -> list[tuple[int, str, list[str]]]:
    """`_agent_prompts.md` から batch 抽出。

    prompt 内に ```json``` 等が ネストされているので、 `## Batch ...` で chunk 分割し、
    各 chunk の最後の ``` を 閉じる、 形式で再構築する。

    Returns:
        list of (batch_index, prompt_text, slide_ids)
    """
    text = prompts_md.read_text(encoding="utf-8")
    # `## Batch N/M` で chunk split (lookbehind 不可、 lookahead で)
    chunks = re.split(r"(?=^##\s+Batch\s+\d+/\d+)", text, flags=re.MULTILINE)
    out: list[tuple[int, str, list[str]]] = []
    head_pat = re.compile(
        r"##\s+Batch\s+(\d+)/\d+:\s+([^→\n]+?)(?:→\s*[^\n]*)?\n+```\n",
    )
    for chunk in chunks:
        m = head_pat.match(chunk.lstrip())
        if not m:
            continue
        bi = int(m.group(1))
        slides_str = m.group(2)
        slide_ids = re.findall(r"slide_\d+", slides_str)
        # 開きフェンス後 〜 chunk 内の最後の ``` までを prompt 本体に
        body_start = m.end()
        rest = chunk.lstrip()[body_start:]
        # 最後の ``` (空行以後しか続かない) を 探す
        last_fence = rest.rfind("\n```")
        if last_fence < 0:
            # ``` で閉じてない (末尾チャンクで途切れている可能性) → そのまま採用
            prompt = rest.strip()
        else:
            prompt = rest[:last_fence]
        out.append((bi, prompt.strip(), slide_ids))
    return out


def _build_feedback_appendix(
    feedback_map: dict[str, str] | None, slide_ids: list[str],
) -> str:
    """retry 時に prompt 末尾に追記する auditor feedback section。

    `feedback_map`: {"slide_10": "...", "slide_13": "..."} の dict。
    対象 batch の slide_ids に含まれるものだけ抽出して追記する。
    """
    if not feedback_map:
        return ""
    relevant = [(sid, feedback_map[sid]) for sid in slide_ids if sid in feedback_map]
    if not relevant:
        return ""
    lines = [
        "",
        "",
        "# 重要: 前回 audit の指摘 (必読、 必ず修正してください)",
        "",
        "前回の出力が seminar-order-auditor で FAIL 判定されました。",
        "下記 slide は特に下記方針で書き直してください。 `row_NN` / `col_NN` 等の",
        "機械的ラベル禁止、 anim も 3 種類以上使い分けること。",
        "",
    ]
    for sid, fb in relevant:
        lines.append(f"## {sid} への指摘")
        lines.append("")
        lines.append(fb)
        lines.append("")
    return "\n".join(lines)


def run_one_batch(
    batch_index: int, prompt: str, slide_ids: list[str], orders_dir: Path,
    model: str = DEFAULT_MODEL, timeout: int = TIMEOUT_PER_BATCH,
    feedback_map: dict[str, str] | None = None,
) -> tuple[int, list[str], list[str], str]:
    """1 batch を副作用なしで生成し、応答取得後に parse → Write。

    Returns:
        (batch_index, success_slide_ids, failed_slide_ids, log_summary)
    """
    t0 = time.time()
    full_prompt = prompt + _build_feedback_appendix(feedback_map, slide_ids)
    try:
        output = run_llm_cli("", full_prompt, model=model, timeout=timeout, cwd=str(ROOT))
    except RuntimeError as exc:
        return (batch_index, [], slide_ids, f"LLM CLI error: {str(exc)[:300]}")
    failed = parse_batch_output(output, orders_dir, slide_ids)
    success = [s for s in slide_ids if s not in failed]
    elapsed = time.time() - t0
    return (
        batch_index, success, failed,
        f"OK in {elapsed:.1f}s / success={len(success)} / failed={len(failed)}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="<post-id>")
    ap.add_argument("--workers", type=int, default=3, help="並列バッチ数 (default 3)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_PER_BATCH, help="1 バッチ秒数 (default 600)")
    ap.add_argument("--only-sids", default="", help="指定 sid (comma 区切り 例: 02,03) を含む batch のみ実行。 個別 slide 改善ボタン用")
    ap.add_argument(
        "--feedback-file", default=None,
        help="audit_orders.py が書く _audit_report.json のパス。 FAIL slide の "
             "feedback_for_orderer を retry prompt に注入する。"
    )
    args = ap.parse_args()

    # feedback_map: {"slide_10": "row_NN ラベルが ...", ...}
    feedback_map: dict[str, str] = {}
    if args.feedback_file:
        try:
            rep = json.loads(Path(args.feedback_file).read_text(encoding="utf-8"))
            for r in rep.get("results", []):
                if r.get("status") == "FAIL" and r.get("feedback_for_orderer"):
                    feedback_map[r["slide_id"]] = r["feedback_for_orderer"]
            print(f"→ feedback 注入: {len(feedback_map)} slides ({sorted(feedback_map.keys())})")
        except Exception as e:
            print(f"WARN: --feedback-file 読み込み失敗 ({e}) → feedback なしで続行", file=sys.stderr)

    topic = args.topic
    # fix: POST_VIDEO_ROOT 尊重 (別 root を使う環境 の orderer 自動化対応)。
    # 旧コードは content_pipeline/posts/ ハードコードで、POST_VIDEO_ROOT で出力先ルートを変えると即 exit=1 になっていた。
    orders_dir = paths.orders_dir(topic)
    prompts_md = orders_dir / "_agent_prompts.md"

    if not prompts_md.exists():
        print(f"ERROR: prompts not found: {prompts_md}", file=sys.stderr)
        print("先に `python -m src.post_video.blob_anim_prepare --topic <topic>` を実行してください。", file=sys.stderr)
        return 1

    batches = extract_batches(prompts_md)
    if not batches:
        print(f"WARN: no batches found in {prompts_md} — 既に全 slide 完了済の可能性", file=sys.stderr)
        return 0

    # --only-sids で 指定 sid を含む batch だけに 絞り込み
    if args.only_sids.strip():
        wanted = {s.strip() for s in args.only_sids.split(",") if s.strip()}
        # batch 内の slide_ids は ["slide_02", ...] 形式 / wanted は "02" 形式
        def _batch_matches(slide_ids: list[str]) -> bool:
            return any(sid.replace("slide_", "") in wanted for sid in slide_ids)
        before_n = len(batches)
        batches = [b for b in batches if _batch_matches(b[2])]
        print(f"→ --only-sids={args.only_sids} で {before_n} → {len(batches)} batch に絞り込み")
        if not batches:
            print("⚠ 指定 sid を含む batch がありません", file=sys.stderr)
            return 0

    print(f"→ {len(batches)} batch(es) を 並列 {args.workers} で LLM CLI 実行")
    for bi, _, slide_ids in batches:
        print(f"  batch {bi}: {','.join(slide_ids)}")

    all_failed: list[str] = []
    all_success: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                run_one_batch, bi, prompt, slide_ids, orders_dir,
                args.model, args.timeout, feedback_map,
            ): bi
            for bi, prompt, slide_ids in batches
        }
        for fut in as_completed(futs):
            bi = futs[fut]
            try:
                _, success, failed, log = fut.result()
                print(f"  [batch {bi}] {log}")
                all_success.extend(success)
                all_failed.extend(failed)
            except Exception as e:
                print(f"  [batch {bi}] EXCEPTION: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"→ 全 batch 完了 in {elapsed:.1f}s / success={len(all_success)} / failed={len(all_failed)}")
    if all_failed:
        print(f"⚠ {len(all_failed)} slides failed: {','.join(all_failed)}")
        print("  → orchestrator は blob_order_rulebased で fallback してください")
    return 0


if __name__ == "__main__":
    sys.exit(main())
