"""seminar-order-auditor の audit → orderer retry → finalize ループ。

設計思想:
  - 「諦めても build は止めない、 ただし諦めた事実を 3 箇所に残す」
    1. work/_audit_report.json の `gave_up_slides` / `gave_up_after_retries`
    2. stdout の "GAVE_UP" ログ
    3. UI バナー (server.py が _audit_report.json を読んで表示)
  - orderer agent の機械的逃げ出力 (row_NN / 全 fade_in) を捕まえて、
    auditor の feedback_for_orderer を prompt 注入して 同 slide を再 spawn。

Usage:
  python -m src.post_video.audit_loop --topic <post-id>
  python -m src.post_video.audit_loop --topic <post-id> --max-retries 2

Exit code:
  0 全 PASS で抜けた / 諦めて build 続行
  2 auditor 自体が動かなかった (CLI エラー等、 build は続行されるべき)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import paths


def run_audit_loop(topic: str, max_retries: int = 2, model: str | None = None) -> int:
    """Returns 0 (success or graceful give-up) or 2 (auditor itself broken)."""
    py = sys.executable
    root = paths.repo_root()
    audit_report_path = paths.work_dir(topic) / "_audit_report.json"

    gave_up_slides: list[str] = []
    audit_cmd_base = [py, "-m", "src.post_video.audit_orders", "--topic", topic]
    if model:
        audit_cmd_base += ["--model", model]

    for attempt in range(max_retries + 1):
        r_audit = subprocess.run(audit_cmd_base, cwd=str(root))
        if r_audit.returncode == 0:
            print(f"[audit_loop] audit PASS (attempt {attempt + 1}/{max_retries + 1})", flush=True)
            return 0
        if r_audit.returncode == 2:
            sys.stderr.write(
                f"[audit_loop] auditor itself failed (rc=2) attempt {attempt + 1}/{max_retries + 1} "
                f"— build 続行 (人間の目視確認推奨)\n"
            )
            return 2
        # rc==1: FAIL あり
        try:
            rep = json.loads(audit_report_path.read_text(encoding="utf-8"))
            fail_sids_full = rep.get("summary", {}).get("fail_slides", [])
            fail_sids_short = [s.replace("slide_", "") for s in fail_sids_full]
        except Exception as e:
            sys.stderr.write(f"[audit_loop] _audit_report.json 読込失敗 ({e}) — build 続行\n")
            return 2

        if not fail_sids_short:
            return 0

        if attempt == max_retries:
            gave_up_slides = list(fail_sids_full)
            print(
                f"[audit_loop] GAVE_UP after {max_retries} retries: {gave_up_slides}\n"
                f"  rule-based fallback の order JSON のまま build 続行。\n"
                f"  人間が UI バナーで諦め slide を確認 → 個別 orderer 改善 ボタンで再挑戦できる。",
                flush=True,
            )
            break

        print(
            f"[audit_loop] FAIL attempt {attempt + 1}/{max_retries + 1} "
            f"→ orderer retry: {fail_sids_full}",
            flush=True,
        )
        sids_arg = ",".join(fail_sids_short)
        r_retry = subprocess.run(
            [py, "-m", "src.post_video.orderer_run",
             "--topic", topic, "--workers", "3",
             "--only-sids", sids_arg,
             "--feedback-file", str(audit_report_path)],
            cwd=str(root),
        )
        if r_retry.returncode != 0:
            sys.stderr.write(
                f"[audit_loop] orderer_run rc={r_retry.returncode} — 次の audit へ進む\n"
            )
        # retry 後は finalize して resolved_blobs を埋め直す
        r_fin = subprocess.run(
            [py, "-m", "src.post_video.blob_anim_finalize", "--topic", topic],
            cwd=str(root),
        )
        if r_fin.returncode != 0:
            sys.stderr.write(
                f"[audit_loop] blob_anim_finalize rc={r_fin.returncode} — gate 抜ける\n"
            )
            return 2

    # 諦め記録を report に注入 (UI バナー表示用)
    if gave_up_slides and audit_report_path.exists():
        try:
            rep = json.loads(audit_report_path.read_text(encoding="utf-8"))
            rep["gave_up_slides"] = gave_up_slides
            rep["gave_up_after_retries"] = max_retries
            rep["gave_up_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            audit_report_path.write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        except Exception as e:
            sys.stderr.write(f"[audit_loop] gave_up 記録失敗 ({e})\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    return run_audit_loop(args.topic, args.max_retries, args.model)


if __name__ == "__main__":
    sys.exit(main())
