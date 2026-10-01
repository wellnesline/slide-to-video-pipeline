"""orderer agent が出した order JSON を resolve(blob_filter → resolved_blobs)。

`blob_anim_prepare.py` で blob_split + 集約 + agent prompts 出力後、
agent ×N で work/orders/slide_<NN>_order.json が揃ったらこれを実行。

Usage:
  python -m src.post_video.blob_anim_finalize --topic <post-id>
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from . import paths

ROOT = paths.repo_root()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    args = ap.parse_args(argv)

    topic = args.topic
    orders_dir = paths.orders_dir(topic)
    blobs_dir = paths.blobs_dir(topic)
    all_blobs = blobs_dir / "all_blobs.json"

    # 7-A-5e: _expected_orders.json が無い既存 post はスキップ(旧動作維持)
    expected_file = orders_dir / "_expected_orders.json"
    if not expected_file.exists():
        print(f"[skip] _expected_orders.json 不在 → blob_anim_finalize をスキップ (topic={topic})")
        return 0

    if not all_blobs.exists():
        sys.stderr.write(f"missing: {all_blobs}\n先に blob_anim_prepare を実行\n")
        return 1

    order_files = sorted(orders_dir.glob("slide_*_order.json"))
    if not order_files:
        sys.stderr.write(f"order JSON が見つかりません: {orders_dir}\n")
        return 1

    print(f"=== blob_anim_finalize (topic={topic}) ===")
    py = sys.executable
    n_ok = 0
    n_fail = 0
    for f in order_files:
        cmd = [
            py, "-m", "src.diagram_blob.blob_animate",
            "--order", str(f),
            "--all-blobs", str(all_blobs),
            "--out-dir", str(orders_dir),
        ]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  {r.stdout.strip().splitlines()[0]}")
            n_ok += 1
        else:
            print(f"  [error] {f.name}: {r.stderr[-300:]}")
            n_fail += 1

    print(f"\n=== done: {n_ok} OK / {n_fail} fail ===")

    # === Step 末尾: order自動 post-process (mapping強化フェーズ) ===
    # subagent の mapping漏れ / 重複blob / 機械的ラベル / 取りこぼし を 機械ルールで補正。
    # blob_anim_finalize が走った直後に必ず実行する (run_all 経由でも 単体実行でも自動)。
    if n_fail == 0:
        print("\n[finalize 末尾] order_merge.py を自動実行 (mapping漏れ補正)")
        from src.post_video import order_merge as _om
        try:
            _om.main(["--topic", topic])
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"  [WARN] order_merge exit={e.code}", file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] order_merge 例外: {e}", file=sys.stderr)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
