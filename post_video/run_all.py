"""<post-id> パイプライン 4 step 統合実行。

順次実行:
  1. synth          (ElevenLabs 合成)
  2. parallel_render (blob anim 並列レンダリング)
  3. build          (字幕 + per-slide clip + concat)
  4. mix_bgm        (BGM 切替)

各 step 失敗で即停止(--continue-on-error で継続)。
特定 step を飛ばすには --skip-XXX。

Usage:
  python -m src.post_video.run_all --topic <post-id>
  python -m src.post_video.run_all --topic <post-id> --workers 8
  python -m src.post_video.run_all --topic <post-id> --skip-synth --skip-render
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from . import paths

ROOT = Path(__file__).resolve().parents[2]


def _run(label: str, cmd: list[str], continue_on_error: bool) -> int:
    print(f"\n{'=' * 60}\n=== {label} ===\n{'=' * 60}", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"\n--- {label}: exit={p.returncode}, elapsed={elapsed:.1f}s ---", flush=True)
    if p.returncode != 0 and not continue_on_error:
        sys.stderr.write(f"\n[run_all] FAIL at {label}, abort.\n")
        sys.exit(p.returncode)
    return p.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=6, help="parallel_render worker 数")
    ap.add_argument("--voice", default=None, help="ElevenLabs voice_id 上書き")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-bgm", action="store_true")
    ap.add_argument("--skip-qa", action="store_true",
                    help="Step 0 の qa_post_video pre-build BLOCK gate を skip (debug 用、 通常使うな)")
    ap.add_argument("--qa-soft", action="store_true",
                    help="Step 0 qa_post_video に --soft を pass-through。 content 系 F* "
                         "(F1/F3/F4/F5/F9/F9-c/F9-d) を WARN 降格して build を最後まで通す。"
                         " client demo / 緊急公開 用途。 ファイル整合性系 (F11/F14) は BLOCK 維持。")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Step 1.5 の verify_synth (Whisper) gate を skip (debug 用)")
    ap.add_argument("--skip-image-check", action="store_true",
                    help="Step 1.8 の slides_raw png 存在チェックを skip (gen_slide_images 完了済みを保証できる場合)")
    ap.add_argument("--auto-gen-images", action="store_true",
                    help="Step 1.8 で png 不足時に gen_slide_images を自動実行 (OpenAI 課金発生、明示オプトイン)")
    # NOTE: 旧 --skip-blob-check は 物理削除(5回の静止画フォールバック事故根治)
    #   緊急時の静止画 escape は別スクリプト経由に隔離する設計予定
    #   Claude セッションが本flag を 提案/復活したら違反扱い
    ap.add_argument("--no-self-heal", action="store_true",
                    help="verify_synth BLOCK 時の self-heal 自動試行を無効化")
    ap.add_argument("--skip-audit", action="store_true",
                    help="Step 1.87 の seminar-order-auditor gate を skip (debug 用)")
    ap.add_argument("--audit-max-retries", type=int, default=2,
                    help="audit FAIL 時の orderer 再 spawn 回数 (default 2)。 "
                         "上限に達したら諦めて _audit_report.json に GAVE_UP を記録、 build は続行")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="step 失敗時に次の step に進む(default は abort)")
    args = ap.parse_args(argv)

    # 過去の事故由来 : Shorts 系を run_all で起動するのは BLOCK (横長 build 事故防止)
    # run_all は本編専用の build.py + mix_bgm.py を呼ぶため、 Shorts topic で起動すると
    # 横長 1920×1080 でレンダされる。 Shorts は build_short.py + mix_bgm_short.py を使う。
    if "/shorts/" in args.topic:
        sys.stderr.write(
            f"[run_all] BLOCK: topic={args.topic} は Shorts 系統。\n"
            f"  `run_all.py` は本編専用 (build.py + mix_bgm.py を呼ぶため横長になる)。\n"
            f"  → 代わりに以下を順次実行:\n"
            f"     python -m src.post_video.synth         --topic {args.topic} --workers 4\n"
            f"     python -m src.post_video.parallel_render --topic {args.topic} --workers 4\n"
            f"     python -m src.post_video.build_short    --topic {args.topic} --workers 3\n"
            f"     python -m src.post_video.mix_bgm_short  --topic {args.topic}\n"
        )
        sys.exit(2)

    py = sys.executable
    cont = args.continue_on_error
    overall_t0 = time.time()

    # Pre-step: slides.json sha sentinel 比較で cache invalidation 警告
    # 監査 1 由来: slides.json 編集後に旧 anim/clip cache が残り
    # 「めちゃくちゃ動画」 が生成される事故対策。 sha 変化を build 開始時に警告。
    # parallel_render.py の content_hash + gen_slide_images.py の content_hash が
    # 個別 sid 単位で invalidation するので、 ここは「人間視認用の警告」 のみ(BLOCK しない)。
    slides_path = paths.slides_json(args.topic)
    sentinel_path = paths.work_dir(args.topic) / "_slides_sha.txt"
    if slides_path.exists():
        cur_sha = hashlib.sha256(slides_path.read_bytes()).hexdigest()[:16]
        prev_sha = sentinel_path.read_text().strip() if sentinel_path.exists() else None
        if prev_sha and prev_sha != cur_sha:
            print(f"\n[run_all] NOTICE: slides.json sha 変更を検出 ({prev_sha} → {cur_sha})", flush=True)
            print(f"  下流 cache (slides_raw / anim / clip) は content_hash で個別に再生成されます", flush=True)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(cur_sha)

    # Step 0: 構造 QA (caption/speech 乖離 + 数字 kana 等を build 前にハードガード)
    # 監査 3 結論: agent 自己申告では BLOCK が bypass され続けたため、
    # build パイプライン入口で BLOCK 検出時は abort する設計に変更。
    # exit 1 = BLOCK あり (abort)、 exit 2 = WARN のみ (build 続行)
    if not args.skip_qa:
        label = "Step 0/4: qa_post_video (pre-build BLOCK gate)"
        if args.qa_soft:
            label += " [soft]"
        print(f"\n{'=' * 60}\n=== {label} ===\n{'=' * 60}", flush=True)
        qa_cmd = [py, "-m", "src.post_video.qa_post_video", "--topic", args.topic]
        if args.qa_soft:
            qa_cmd.append("--soft")
        r = subprocess.run(qa_cmd, cwd=str(ROOT))
        if r.returncode == 1:
            sys.stderr.write(f"\n[run_all] qa_post_video BLOCK at topic={args.topic}, abort.\n")
            sys.exit(1)

    if not args.skip_synth:
        cmd = [py, "-m", "src.post_video.synth", "--topic", args.topic]
        if args.voice:
            cmd += ["--voice", args.voice]
        _run("Step 1/4: synth", cmd, cont)

    # Step 1.5: verify_synth (Whisper で実音声 vs speech 比較)
    # 監査 1 結論: synth wav cache skip bug + ElevenLabs 誤読 の合わせ技で
    # 「字幕と音声が違う」 事故が連発。 build パイプ自動 gate 化で再発防止(過去の事故由来)。
    # exit 1 = 致命的 mismatch (ratio < 0.5 ≥ 1 件 or mismatch 率 > 15%) → abort
    # --self-heal 1 周自動試行: 致命的 line の mp3 cache 削除 → 再合成 → 再 verify
    if not args.skip_verify:
        print(f"\n{'=' * 60}\n=== Step 1.5/4: verify_synth (Whisper TTS sanity gate) ===\n{'=' * 60}", flush=True)
        for attempt in (1, 2):
            r = subprocess.run(
                [py, "-m", "src.post_video.verify_synth",
                 "--topic", args.topic, "--strict"],
                cwd=str(ROOT),
            )
            if r.returncode == 0:
                break
            if attempt == 1 and not args.no_self_heal:
                sys.stderr.write("\n[run_all] verify_synth BLOCK → --self-heal で 1 周自動再合成試行\n")
                r2 = subprocess.run(
                    [py, "-m", "src.post_video.synth",
                     "--topic", args.topic, "--self-heal"],
                    cwd=str(ROOT),
                )
                if r2.returncode != 0:
                    sys.stderr.write("[run_all] self-heal 失敗、 abort.\n")
                    sys.exit(1)
                # 2 周目 verify_synth へループ
                continue
            sys.stderr.write(f"\n[run_all] verify_synth BLOCK at topic={args.topic}, abort. kana_dict 整備 or speech 修正後に再実行。\n")
            sys.exit(1)

    # Step 1.8: slides_raw png 存在チェック gate
    # 過去の事故由来: gen_slide_images 未実行のまま run_all を起動すると
    # parallel_render がサイレント全スキップ → build 失敗するフェイルモードを排除。
    # --skip-render の場合は parallel_render 自体を呼ばないため gate 不要。
    if not args.skip_render and not args.skip_image_check:
        print(f"\n{'=' * 60}\n=== Step 1.8/4: slides_raw png 存在チェック gate ===\n{'=' * 60}", flush=True)
        slides_path_check = paths.slides_json(args.topic)
        pending_sids: list[str] = []
        if slides_path_check.exists():
            slides_data = json.loads(slides_path_check.read_text(encoding="utf-8"))
            raw_dir = paths.slides_raw_dir(args.topic)
            # slides.py load_slides の流儀に揃える:
            #   - string entry(SECTION マーカー等)は skip、 real_idx インクリメントしない
            #   - dict entry は shared_from 含めて real_idx インクリメント(sid 連番採番)
            #   - shared_from エントリは raw.png を shared_slide_dir に持つので slides_raw 対象外
            real_idx = 0
            for slide in slides_data.get("slides", []):
                if not isinstance(slide, dict):
                    continue
                real_idx += 1
                sid = f"{real_idx:02d}"
                if "shared_from" in slide:
                    continue
                png_path = raw_dir / f"{sid}.png"
                if not png_path.exists():
                    pending_sids.append(sid)
        if pending_sids:
            if args.auto_gen_images:
                sys.stderr.write(
                    f"\n[run_all] Step 1.8: {len(pending_sids)} 件の slides_raw png が不在 → "
                    f"--auto-gen-images が指定されているため gen_slide_images を自動実行します\n"
                )
                r_gen = subprocess.run(
                    [py, "-m", "src.post_video.gen_slide_images", "--topic", args.topic],
                    cwd=str(ROOT),
                )
                if r_gen.returncode != 0:
                    sys.stderr.write(f"\n[run_all] gen_slide_images 失敗 (exit={r_gen.returncode})、 abort.\n")
                    sys.exit(r_gen.returncode)
                print(f"\n[run_all] gen_slide_images 完了、 pipeline を継続します", flush=True)
            else:
                sys.stderr.write(
                    f"\n[run_all] BLOCK: slides_raw png が {len(pending_sids)} 件不在: {pending_sids}\n"
                    f"  先に以下を実行してください:\n"
                    f"    python -m src.post_video.gen_slide_images --topic {args.topic}\n"
                    f"  画像生成を run_all 内で自動実行したい場合は --auto-gen-images を付けてください。\n"
                    f"  チェックを skip したい場合(gen_slide_images 完了済み保証がある場合)は "
                    f"--skip-image-check を付けてください。\n"
                )
                sys.exit(1)
        else:
            print(f"\n[run_all] Step 1.8: slides_raw png 全件確認済み (shared 除外後チェック OK)", flush=True)

    # Step 1.85: resolved_blobs 欠落自動 finalize gate
    # blob_finalize_guard spike 由来: orderer agent 実行後に
    # blob_anim_finalize を実行し忘れると resolved_blobs が欠落したまま parallel_render が
    # 走り、 blob_renderer が空リスト → reveal 何もしない → base_img のみの動画が生成される。
    # parallel_render は ok=N で正常完了と報告するため人間が気付けない(過去の事故)。
    # 対策: resolved_blobs 欠落を検出したら blob_anim_finalize を自動実行する。
    # _expected_orders.json が無い旧 post は finalize 自体が skip を返すため副作用なし。
    # --skip-render の場合は parallel_render 自体を呼ばないため gate 不要。
    if not args.skip_render:
        _step185_orders_dir = paths.orders_dir(args.topic)
        _expected_file = _step185_orders_dir / "_expected_orders.json"
        if _expected_file.exists():
            print(f"\n{'=' * 60}\n=== Step 1.85/4: resolved_blobs 欠落チェック gate ===\n{'=' * 60}", flush=True)
            _order_files = sorted(_step185_orders_dir.glob("slide_*_order.json"))
            _unresolved_sids: list[str] = []
            for _of in _order_files:
                try:
                    _od = json.loads(_of.read_text(encoding="utf-8"))
                    _groups = _od.get("groups", [])
                    # blob_filter が非空かつ resolved_blobs が欠落しているグループが 1 つでもある slide
                    _needs_finalize = any(
                        g.get("blob_filter") and not g.get("resolved_blobs")
                        for g in _groups
                    )
                    if _needs_finalize:
                        _unresolved_sids.append(_of.stem)
                except Exception:
                    pass  # JSON 読込失敗は無視(Step 1.9 で別途検出)
            if _unresolved_sids:
                print(
                    f"[run_all] Step 1.85: {len(_unresolved_sids)} 件の order JSON に "
                    f"resolved_blobs が欠落: {_unresolved_sids}\n"
                    f"  → blob_anim_finalize を自動実行します (API 課金なし / pure local 処理)",
                    flush=True,
                )
                _r_finalize = subprocess.run(
                    [py, "-m", "src.post_video.blob_anim_finalize", "--topic", args.topic],
                    cwd=str(ROOT),
                )
                if _r_finalize.returncode != 0:
                    sys.stderr.write(
                        f"\n[run_all] blob_anim_finalize 失敗 (exit={_r_finalize.returncode})、 abort.\n"
                        f"  orderer agent の出力 order JSON が壊れている可能性があります。\n"
                        f"  orders ディレクトリを確認してください: {_step185_orders_dir}\n"
                    )
                    sys.exit(_r_finalize.returncode)
                print(f"\n[run_all] Step 1.85: blob_anim_finalize 完了、 pipeline を継続します", flush=True)
            else:
                print(f"\n[run_all] Step 1.85: resolved_blobs 全件確認済み (finalize 済み OK)", flush=True)

    # Step 1.87: seminar-order-auditor gate (機械的逃げ出力 hard gate)
    # 追加: 過去の複数の回で blob_order_rulebased が
    # row_NN + 全 fade_in の機械出力を黙って書き込み、 build が「動くけど雑な動画」 を量産する
    # 事故の根治。 5軸監査 (F1 機械ラベル / F2 anim 単一 / F3 group数 / F4 言語整合 / F5 filter形式)
    # で FAIL なら orderer を最大 N 回 retry (auditor の feedback_for_orderer を prompt 注入)。
    # 上限到達後も FAIL なら諦めて _audit_report.json に GAVE_UP を記録 (build は続行)。
    # 諦め slide は UI に黄色バナー表示 + 人間が「orderer 改善」 ボタンで個別再挑戦できる。
    if not args.skip_render and not args.skip_audit:
        print(f"\n{'=' * 60}\n=== Step 1.87/4: seminar-order-auditor gate ===\n{'=' * 60}", flush=True)
        r_audit_loop = subprocess.run(
            [py, "-m", "src.post_video.audit_loop",
             "--topic", args.topic,
             "--max-retries", str(args.audit_max_retries)],
            cwd=str(ROOT),
        )
        # audit_loop は 0 / 2 のみ返す (両方 build 続行)。 1 を返さないので blocking しない。
        if r_audit_loop.returncode not in (0, 2):
            sys.stderr.write(
                f"[run_all] Step 1.87: audit_loop が予期しない rc={r_audit_loop.returncode} を返した — build 続行\n"
            )

    # Step 1.9: blob deps gate (labels.npz + order JSON 存在チェック)
    # v2 事故根治: blob deps が揃っていないまま parallel_render を
    # 起動すると全 slide が "skip: deps missing" でサイレント完走し、 build で anim なし
    # 静止画フォールバックのまま final.mp4 が生成される三連事故(v1/v2/v2c)を根治。
    # --skip-render の場合は parallel_render 自体を呼ばないため gate 不要。
    # --skip-blob-check 物理削除、 本 gate は常時ON、 緊急 escape は別スクリプトへ。
    if not args.skip_render:
        print(f"\n{'=' * 60}\n=== Step 1.9/4: blob deps gate (labels.npz + order JSON) ===\n{'=' * 60}", flush=True)
        slides_path_blob = paths.slides_json(args.topic)
        total_post = 0  # スコープ外参照に備えて先に 0 で初期化
        pending_blob_sids: list[str] = []
        if slides_path_blob.exists():
            try:
                slides_data_blob = json.loads(slides_path_blob.read_text(encoding="utf-8"))
                blobs_base = paths.blobs_dir(args.topic) if hasattr(paths, "blobs_dir") else None
                orders_base = paths.orders_dir(args.topic) if hasattr(paths, "orders_dir") else None
                # slides.py load_slides と同じ real_idx 連番採番
                real_idx = 0
                for slide in slides_data_blob.get("slides", []):
                    if not isinstance(slide, dict):
                        continue
                    real_idx += 1
                    sid = f"{real_idx:02d}"
                    if "shared_from" in slide:
                        continue  # shared slide は blob 不要
                    total_post += 1
                    labels_ok = False
                    order_ok = False
                    if blobs_base is not None:
                        labels_ok = (blobs_base / f"slide_{sid}" / "labels.npz").exists()
                    if orders_base is not None:
                        order_ok = (orders_base / f"slide_{sid}_order.json").exists()
                    if not (labels_ok and order_ok):
                        pending_blob_sids.append(sid)
            except Exception as _e:
                print(f"\n[run_all] Step 1.9: slides.json 読込失敗 ({_e})、 blob deps チェックをスキップします", flush=True)
                total_post = 0  # 安全パス: チェック不能時は BLOCK しない
        if total_post >= 3 and len(pending_blob_sids) > total_post * 0.5:
            sys.stderr.write(
                f"\n[run_all] BLOCK: blob deps が {len(pending_blob_sids)}/{total_post} 件不在 "
                f"(>{50:.0f}%): {pending_blob_sids}\n"
                f"  parallel_render を起動するとサイレント全スキップ → anim なし静止画フォールバックになります。\n"
                f"  以下の手順で deps を生成してから再実行してください:\n"
                f"    1. python -m src.post_video.blob_anim_prepare --topic {args.topic}\n"
                f"    2. orderer agent ×N 並列 spawn (orders/*.json 生成)\n"
                f"    3. python -m src.post_video.blob_anim_finalize --topic {args.topic}\n"
                f"    4. python -m src.post_video.run_all --topic {args.topic} [その他オプション]\n"
                f"  ※ --skip-blob-check は 物理削除(5回の静止画事故根治)、 緊急時は 人間に エスカレ。\n"
            )
            sys.exit(1)
        else:
            print(f"\n[run_all] Step 1.9: blob deps OK "
                  f"(post_individual={total_post}, pending={len(pending_blob_sids)})", flush=True)

    if not args.skip_render:
        cmd = [py, "-m", "src.post_video.parallel_render",
               "--topic", args.topic, "--workers", str(args.workers)]
        # --skip-blob-check 削除、 parallel_render 側にも伝播しない
        _run("Step 2/4: parallel_render", cmd, cont)

    if not args.skip_build:
        cmd = [py, "-m", "src.post_video.build", "--topic", args.topic]
        _run("Step 3/4: build", cmd, cont)

    if not args.skip_bgm:
        cmd = [py, "-m", "src.post_video.mix_bgm", "--topic", args.topic]
        _run("Step 4/4: mix_bgm", cmd, cont)

    overall_elapsed = time.time() - overall_t0
    final_mp4 = paths.final_mp4(args.topic)
    print(f"\n{'=' * 60}")
    print(f"=== ALL DONE (total {overall_elapsed:.1f}s) ===")
    print(f"{'=' * 60}")
    if final_mp4.exists():
        print(f"\nfinal: {final_mp4}")
        print(f"  size: {final_mp4.stat().st_size // (1024*1024)} MB")
    else:
        print(f"\nfinal mp4 が見つかりません: {final_mp4}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
