"""order-auditor agent を spawn して order JSON の品質を監査する。

過去の複数の回で見つかった (slide_13/14/16 など) ような「機械的逃げ出力」
(row_NN / col_NN ラベル + 全 fade_in) を 5軸 hard gate で検知する。

シーケンス:
  1. orders/_expected_orders.json から監査対象 slide_ids を取得
  2. order-auditor.md の指示を選択中の LLM CLI に読ませて1回生成
  3. 出力の `=== audit_report ===` ブロックを parse
  4. work/_audit_report.json に書き出し
  5. exit code:
       0  全 PASS / WARN のみ
       1  FAIL あり (run_all.py の retry trigger)
       2  parse 失敗 / CLI エラー (運用判断要)

Usage:
  python -m src.post_video.audit_orders --topic <post-id>
  python -m src.post_video.audit_orders --topic <post-id> --slide-ids slide_10,slide_13
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

from . import paths
from src.claude_cli import run_llm_cli

ROOT = paths.repo_root()
DEFAULT_MODEL = "claude-sonnet-4-5"
TIMEOUT_SEC = 600

AUDITOR_AGENT_MD = ROOT / ".claude" / "agents" / "order-auditor.md"


def _load_expected_slide_ids(orders_dir: Path) -> list[str]:
    expected_file = orders_dir / "_expected_orders.json"
    if not expected_file.exists():
        return []
    data = json.loads(expected_file.read_text(encoding="utf-8"))
    sids = data.get("slide_ids", [])
    return [f"slide_{sid}" for sid in sids]


def _build_prompt(topic: str, slide_ids: list[str], orders_dir: Path) -> str:
    """auditor 用 prompt を組み立て。

    order JSON 本体を Read させると agent 起動コストが大きいので、
    要点 (groups の label / anim / blob_filter) のみ prompt に埋め込む。
    """
    lines: list[str] = []
    lines.append("あなたは order-auditor agent として動作します。")
    lines.append(
        f"完全な agent 定義は `{AUDITOR_AGENT_MD.relative_to(ROOT)}` を Read で読み、 "
        "その指示に厳密に従ってください。"
    )
    lines.append("")
    lines.append(f"- topic: `{topic}`")
    lines.append(f"- 監査対象 slide: {slide_ids}")
    lines.append("")
    lines.append("# 監査対象 order JSON 一覧")
    lines.append("")

    for full_sid in slide_ids:
        order_path = orders_dir / f"{full_sid}_order.json"
        if not order_path.exists():
            lines.append(f"## {full_sid}")
            lines.append("")
            lines.append("(order JSON 不在 → F3 FAIL 扱い)")
            lines.append("")
            continue
        try:
            data = json.loads(order_path.read_text(encoding="utf-8"))
        except Exception as e:
            lines.append(f"## {full_sid}")
            lines.append("")
            lines.append(f"(JSON parse error: {e} → F5 FAIL 扱い)")
            lines.append("")
            continue
        groups = data.get("groups", [])
        generated_by = data.get("_generated_by", "(unknown)")
        lines.append(f"## {full_sid}")
        lines.append("")
        lines.append(f"- `_generated_by`: `{generated_by}`")
        lines.append(f"- groups: {len(groups)}")
        lines.append("```json")
        # 監査軸は label / anim / blob_filter のみ必要、 resolved_blobs は省略
        slim = [
            {
                "label": g.get("label", ""),
                "anim": g.get("anim", ""),
                "blob_filter": g.get("blob_filter", ""),
            }
            for g in groups
        ]
        lines.append(json.dumps(slim, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("# 出力指示")
    lines.append("")
    lines.append(
        "監査結果を `=== audit_report ===` ... `=== end_audit_report ===` で囲み、 "
        "中身は agent 定義の §4 出力形式に従う JSON 1件としてください。"
    )
    lines.append("FAIL slide には必ず `feedback_for_orderer` フィールドを書くこと。")
    return "\n".join(lines)


def _run_auditor(prompt: str, model: str, timeout: int) -> tuple[int, str, str]:
    """副作用なしで auditor 応答を生成する。

    Returns: (returncode, stdout, stderr)
    """
    try:
        text = run_llm_cli("", prompt, model=model, timeout=timeout, cwd=str(ROOT))
        return (0, text, "")
    except RuntimeError as exc:
        return (2, "", str(exc))


def _parse_report(stdout: str) -> dict | None:
    """`=== audit_report ===` ブロックから JSON 抜き出し。"""
    m = re.search(
        r"===\s*audit_report\s*===\s*\n(.*?)\n===\s*end_audit_report\s*===",
        stdout, re.DOTALL,
    )
    if not m:
        # block fence ナシでも JSON 単体で返るケースを救済
        try:
            return json.loads(stdout.strip())
        except Exception:
            return None
    body = m.group(1).strip()
    # ```json ... ``` の fence 除去
    if body.startswith("```"):
        body_lines = body.split("\n")
        end = len(body_lines)
        if body_lines[-1].startswith("```"):
            end -= 1
        body = "\n".join(body_lines[1:end]).strip()
    try:
        return json.loads(body)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--slide-ids", default="",
                    help="カンマ区切り (例: slide_10,slide_13)。 省略時は _expected_orders.json 全件")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_SEC)
    ap.add_argument("--out", default=None,
                    help="出力先 (default: <topic>/work/_audit_report.json)")
    args = ap.parse_args(argv)

    topic = args.topic
    orders_dir = paths.orders_dir(topic)
    work_dir = paths.work_dir(topic)
    if not orders_dir.exists():
        sys.stderr.write(f"orders dir 不在: {orders_dir}\n")
        return 2

    if args.slide_ids.strip():
        slide_ids = [s.strip() for s in args.slide_ids.split(",") if s.strip()]
    else:
        slide_ids = _load_expected_slide_ids(orders_dir)
    if not slide_ids:
        print("[audit_orders] 監査対象 slide なし (skip)")
        return 0

    print(f"[audit_orders] topic={topic} target={len(slide_ids)} slides")
    prompt = _build_prompt(topic, slide_ids, orders_dir)
    rc, stdout, stderr = _run_auditor(prompt, args.model, args.timeout)
    if rc != 0:
        sys.stderr.write(f"[audit_orders] LLM CLI rc={rc}\n{stderr[:400]}\n")
        return 2

    report = _parse_report(stdout)
    if report is None:
        sys.stderr.write(
            "[audit_orders] auditor 出力 parse 失敗。 stdout 先頭:\n"
            f"{stdout[:600]}\n"
        )
        return 2

    # timestamp / topic を保証
    report.setdefault("topic", topic)
    report.setdefault("audited_at", _dt.datetime.now().isoformat(timespec="seconds"))

    out_path = Path(args.out) if args.out else work_dir / "_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report.get("summary", {})
    fail_slides = summary.get("fail_slides", [])
    print(
        f"[audit_orders] total={summary.get('total', '?')} "
        f"pass={summary.get('pass', '?')} warn={summary.get('warn', '?')} "
        f"fail={summary.get('fail', '?')}"
    )
    print(f"[audit_orders] report → {out_path.relative_to(ROOT)}")
    if fail_slides:
        print(f"[audit_orders] FAIL slides: {fail_slides}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
