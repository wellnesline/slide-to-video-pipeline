"""<post-id> の post 個別 slide について blob 分離 + all_blobs 集約 + agent spawn 指示出力。

シーケンス:
  1. slides_raw/<NN>.png を `slide_<NN>_raw.png` 命名で tempdir に複製(blob_split 仕様)
  2. blob_split.py を全 post 個別 slide に実行 → labels.npz / blobs.json
  3. all_blobs.json 集約
  4. orderer agent の **5-slide バッチ版 spawn プロンプト** をファイル出力
     (複数 slide を 1 回の spawn にまとめて spawn 数を削減)
     (実際の spawn は Claude セッション内で並列実行)

shared slide はスキップ(shared プールに既に blobs/orders 完備)。

Usage:
  python -m src.post_video.blob_anim_prepare --topic <post-id> [--dilate 2]

出力:
  <topic>/work/blobs/slide_<NN>/{blobs.json, labels.npz}
  <topic>/work/blobs/all_blobs.json
  <topic>/work/orders/_agent_prompts.md  ← Claude セッション内で spawn する用
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import paths
from .slides import load_slides

ROOT = paths.repo_root()

# 1 バッチあたりの最大 slide 数。 mapping強化版で 1 に変更:
#   batch=1 集中度UP + raw + overlay + 拡張JSON を Read で 3点突合させるため、
#   1 spawn = 1 slide とする。 並列度は orderer_run の workers で調整 (default 3)。
BATCH_SIZE = 1


def _split_one(args: tuple) -> str:
    sid, raw, blobs_root, dilate, py, root = args
    if not raw.exists():
        return f"  [skip] {sid}: raw png 不在 {raw}"
    with tempfile.TemporaryDirectory(prefix=f"blob_split_{sid}_") as tmpd:
        tmp_raw = Path(tmpd) / f"slide_{sid}_raw.png"
        shutil.copy2(raw, tmp_raw)
        cmd = [py, "-m", "src.diagram_blob.blob_split",
               "--raw", str(tmp_raw),
               "--blobs-dir", str(blobs_root),
               "--dilate", str(dilate),
               "--absorb-faint"]
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if r.returncode != 0:
            return f"  [error] {sid}: blob_split failed\n{r.stderr[-500:]}"
        n_line = r.stdout.strip().splitlines()
        n_blobs = n_line[0].split(":")[-1].strip() if n_line else "?"
        return f"  [split] slide_{sid}: {n_blobs} blobs"


def _generate_batch_prompt(
    batch: list[dict],
    blobs_root: Path,
    slides_raw_dir: Path,
    orders_dir: Path,
    root: Path,
    topic: str = "",
) -> str:
    """単一 slide 用の mapping強化 prompt 生成 (verifier採択)。

    BATCH_SIZE=1 なので batch には 1 slide のみ。 強化ポイント:
      - raw PNG + blob ID overlay PNG + 拡張 blob JSON (neighbors / row_neighbors /
        contained_by 付き) を **必ず 3点 Read** させる
      - row_neighbors を 推移的に拾い直し
      - セルフチェック 5項目 出力前 必須
    """
    s = batch[0]
    sid = s["id"]
    raw_path = s["raw_path"]
    overlay_path = paths.blobs_overlay_dir(topic) / f"slide_{sid}_overlay.png"
    blobs_json_path = blobs_root / f"slide_{sid}" / "blobs.json"
    try:
        raw_rel = raw_path.relative_to(root)
    except ValueError:
        raw_rel = raw_path
    try:
        overlay_rel = overlay_path.relative_to(root)
    except ValueError:
        overlay_rel = overlay_path
    try:
        json_rel = blobs_json_path.relative_to(root)
    except ValueError:
        json_rel = blobs_json_path

    lines: list[str] = []
    lines.append("あなたは orderer agent として動作します。")
    lines.append("完全な agent 定義は `.claude/agents/orderer.md` を Read で読み、 その指示に厳密に従ってください。")
    lines.append("")
    lines.append(f"single slide spawn (batch=1) mode。 担当: slide_{sid}")
    lines.append("")
    lines.append("# 強化指示 (mapping精度向上 verifier採択)")
    lines.append("")
    lines.append("必ず 3 ファイル Read:")
    lines.append(f"1. raw PNG: `{raw_rel}`")
    lines.append(f"2. overlay PNG: `{overlay_rel}` (各 blob ID を 赤地白文字で 視覚表示)")
    lines.append(f"3. enriched blob JSON: `{json_rel}` (`neighbors` / `row_neighbors` / `contained_by` フィールド付き)")
    lines.append("")
    lines.append("# mapping 原則 (最重要)")
    lines.append("- ある blob が「特定の文字列の一部」 と認識されたら、 その blob の `row_neighbors` を **推移的に全部** 同じ group に入れる")
    lines.append("  - 例: blob A が「動画」 と認識 → row_neighbors=[B, C] → B と C も同じ group。 さらに B の row_neighbors=[D, E] → D, E も同じ group (推移的に文字列全体の blob を 集める)")
    lines.append("- 枠 group には `contained_by` で指す大 blob と その枠内の主要要素を含める")
    lines.append("")
    lines.append("# 出力 (ブロック区切り 1個の order JSON)")
    lines.append("```")
    lines.append(f"=== slide_{sid}_order.json ===")
    lines.append(f'{{"diagram_id": "slide_{sid}", "groups": [{{"label": "...", "anim": "wipe_lr|wipe_tb|scale_pop|bounce_in|fade_in", "blob_filter": "manual:00,..."}}]}}')
    lines.append("```")
    lines.append("")
    lines.append("# 出力前 セルフチェック (必須)")
    lines.append("1. label が示す範囲の blob を `row_neighbors` で 推移的に拾い直し 漏れがないか")
    lines.append("2. 全 blob (area > 1000) が 何らかの group に含まれているか")
    lines.append("3. 機械的ラベル (`row_NN` / `col_NN` / `group_NN`) を使っていないか (絶対禁止)")
    lines.append("4. 同じ blob ID が 複数 group に重複登場していないか")
    lines.append("5. anim 最低 3 種類 使い分けているか")
    lines.append("")
    lines.append(f"完了後 `DONE: slide_{sid} processed` と 1 行報告してください。")

    return "\n".join(lines)


def _parse_and_write_batch_output(
    output_text: str,
    orders_dir: Path,
    batch_slide_ids: list[str],
) -> list[str]:
    """バッチ出力から各 slide の order JSON をパースして Write する。

    parse 失敗した slide は WARN を出力し、呼び出し元(orchestrator)が
    単体 spawn フォールバックを行う。

    Returns:
        failed_slide_ids: parse/write に失敗した slide_id のリスト
    """
    failed: list[str] = []

    # Pattern: === slide_XX_order.json === followed by JSON block
    pattern = r"=== (slide_\d+_order\.json) ===\s*\n([\s\S]*?)(?====|DONE:|$)"
    matches = re.findall(pattern, output_text)

    parsed: dict[str, dict] = {}
    for filename, json_text in matches:
        slide_id = filename.replace("_order.json", "")
        json_text = json_text.strip()
        # ```json ... ``` ラップを除去
        if json_text.startswith("```"):
            json_lines = json_text.split("\n")
            end = len(json_lines)
            if json_lines[-1].startswith("```"):
                end = len(json_lines) - 1
            json_text = "\n".join(json_lines[1:end])
        try:
            order_data = json.loads(json_text.strip())
            parsed[slide_id] = order_data
        except json.JSONDecodeError as e:
            print(f"  [WARN] parse failed for {slide_id}: {e}", file=sys.stderr)
            print(f"  raw: {json_text[:200]}", file=sys.stderr)
            failed.append(slide_id)

    # 期待 slide のうち parse できなかったものを追加
    for sid in batch_slide_ids:
        if sid not in parsed and sid not in failed:
            print(f"  [WARN] {sid} not found in batch output — フォールバック必要", file=sys.stderr)
            failed.append(sid)

    # Write
    for slide_id, order_data in parsed.items():
        out_path = orders_dir / f"{slide_id}_order.json"
        out_path.write_text(json.dumps(order_data, ensure_ascii=False, indent=2))
        print(f"  [write] {out_path.relative_to(ROOT)}")

    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4,
                    help="blob_split 並列数(default 4)")
    ap.add_argument("--auto-grouping", action="store_true",
                    help="auto_grouping (v5d) を有効化 (default OFF = subagent 経路)。 default OFF に反転、 旧挙動に戻すため")
    ap.add_argument("--overwrite-orders", action="store_true",
                    help="既存 order JSON があっても auto_grouping で上書きする (default: skip)。 --auto-grouping と併用前提")
    args = ap.parse_args(argv)

    topic = args.topic
    blobs_root = paths.blobs_dir(topic)
    blobs_root.mkdir(parents=True, exist_ok=True)
    orders_dir = paths.orders_dir(topic)
    orders_dir.mkdir(parents=True, exist_ok=True)

    slides = load_slides(topic)
    targets = [s for s in slides if s["shared_id"] is None]
    all_targets = list(targets)  # marker (_expected_orders.json) 用に元 targets を保持
    slides_raw_dir = targets[0]["raw_path"].parent if targets else blobs_root
    print(f"=== blob_anim_prepare (topic={topic}, dilate={args.dilate}, workers={args.workers}) ===")
    print(f"  total: {len(slides)} slides, post 個別: {len(targets)} slides")

    # === Step 1+2: 並列 blob_split ===
    py = sys.executable
    work_args = [(s["id"], s["raw_path"], blobs_root, args.dilate, py, ROOT) for s in targets]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(_split_one, a) for a in work_args]):
            print(f.result(), flush=True)

    # === Step 3: all_blobs.json 集約 ===
    agg = {}
    for sub in sorted(blobs_root.glob("slide_*/")):
        j = sub / "blobs.json"
        if j.exists():
            agg[sub.name] = json.loads(j.read_text()).get("blobs", [])
    all_blobs_path = blobs_root / "all_blobs.json"
    all_blobs_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"  [agg] all_blobs.json: {len(agg)} slides, {sum(len(v) for v in agg.values())} blobs")

    # === Step 3.4: mapping強化 前処理 (verifier採択) ===
    # (a) 空間グラフ (neighbors / row_neighbors / contained_by) を blobs.json に機械付与
    # (b) blob ID overlay PNG を生成して subagent に視覚補助として渡す
    print("\n[Step 3.4] mapping強化 前処理 (空間グラフ + overlay PNG)")
    from src.post_video import preprocess_blobs as _pp
    from src.post_video import generate_overlay as _gv
    _pp.main(["--topic", topic])
    _gv.main(["--topic", topic])
    # 前処理で agg が古いままだと all_blobs.json と blobs.json が乖離するので再集約
    agg = {}
    for sub in sorted(blobs_root.glob("slide_*/")):
        j = sub / "blobs.json"
        if j.exists():
            agg[sub.name] = json.loads(j.read_text()).get("blobs", [])
    all_blobs_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    # === Step 3.5: v5d 機械グルーピング (default OFF、 --auto-grouping で有効化) ===
    # auto_grouping が order JSON を書いた slide は subagent prompt から除外される。
    # default OFF に反転 (user 判定: アニメ動作が subagent 版より劣るため旧挙動に戻す)
    auto_generated_ids: set[str] = set()
    if args.auto_grouping:
        from src.diagram_blob.auto_grouping import build_order
        from PIL import Image as _Image
        print("\n[Step 3.5] auto_grouping (v5d, opt-in)")
        for s in targets:
            sid = s["id"]
            order_path = orders_dir / f"slide_{sid}_order.json"
            if order_path.exists() and not args.overwrite_orders:
                continue
            blobs_data = agg.get(f"slide_{sid}", [])
            if not blobs_data:
                continue
            with _Image.open(s["raw_path"]) as im:
                W, H = im.size
            try:
                order = build_order(s["raw_path"], blobs_data, W, H)
            except Exception as e:
                print(f"  [WARN] auto_grouping failed slide_{sid}: {e}", file=sys.stderr)
                continue
            # <post-id> は slides_raw/NN.png 命名で raw_png.stem = "NN" になる。
            # build_order が diagram_id=NN を入れるが、 post pipeline は "slide_NN" を期待するので強制上書き。
            # これがないと finalize の blob_animate.write_order が NN_order.json (slide_ なし)
            # に書き出し、 parallel_render が slide_NN_order.json (resolved_blobs 無し) を
            # 読んでアニメが空になる事故が発生する (過去の運用で実測)。
            order["diagram_id"] = f"slide_{sid}"
            order_path.write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n")
            auto_generated_ids.add(sid)
            print(f"  [auto] slide_{sid}: route={order.get('_route', '?')} groups={len(order['groups'])}")
        # auto_grouping で order を作った slide は subagent prompt から除外
        if auto_generated_ids:
            targets = [s for s in targets if s["id"] not in auto_generated_ids]
            print(f"  [auto] {len(auto_generated_ids)} slides bypassed subagent")

    # === Step 4: 5-slide バッチ版 agent spawn プロンプト出力(spawn 数を削減) ===
    batches = [targets[i:i+BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
    prompts_md = orders_dir / "_agent_prompts.md"
    lines = [
        f"# {topic} 用 orderer agent spawn プロンプト(バッチ5版)",
        "",
        "**手順**:",
        f"1. Claude セッション内で以下の各バッチプロンプトを Agent ツール(general-purpose subagent_type)に渡して並列 spawn",
        f"2. 全 {len(batches)} バッチを並列実行(1 バッチ = 最大 {BATCH_SIZE} slides)",
        "3. 各 agent は `=== slide_NN_order.json ===` ブロック区切りで返す",
        "4. parse 失敗 slide は単体 spawn フォールバック",
        "5. 完了後 `python -m src.post_video.blob_anim_finalize --topic <topic>` で blob_animate resolve",
        "",
        "---",
        "",
    ]
    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        slide_labels = ", ".join(f"slide_{s['id']}" for s in batch)
        # 完了済みチェック
        all_done = all((orders_dir / f"slide_{s['id']}_order.json").exists() for s in batch)
        status = "✓ 完了" if all_done else "→ 未生成"
        lines.append(f"## Batch {batch_num}/{len(batches)}: {slide_labels} {status}")
        lines.append("")
        lines.append("```")
        prompt = _generate_batch_prompt(batch, blobs_root, slides_raw_dir, orders_dir, ROOT, topic)
        lines.append(prompt)
        lines.append("```")
        lines.append("")

    prompts_md.write_text("\n".join(lines))
    print(f"  [prompts] agent spawn 用プロンプト({len(batches)} batches) → {prompts_md.relative_to(ROOT)}")

    # 7-A-5e: _expected_orders.json を生成(blob_anim_finalize のスキップ判定用マーカー)
    # auto_grouping で事前生成した slide も含む全 post 個別 slide を記録する
    expected = {"slide_ids": [s["id"] for s in all_targets]}
    expected_path = orders_dir / "_expected_orders.json"
    expected_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2))
    print(f"  [marker] _expected_orders.json → {expected_path.relative_to(ROOT)}")

    print("\n=== done ===")
    print(f"次:")
    print(f"  1. {prompts_md.relative_to(ROOT)} を Claude セッションで開いて agent ×{len(batches)} 並列 spawn")
    print(f"  2. python -m src.post_video.blob_anim_finalize --topic {topic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
