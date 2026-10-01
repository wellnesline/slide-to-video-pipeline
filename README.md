# slide-to-video-pipeline

Pipeline that turns a slide script (`slides.json`) into a narrated explainer video: TTS, animated diagram reveals, subtitles, BGM mixing and pre-build QA.

台本（`slides.json`）から、ナレーション・図解の段階表示アニメ・字幕・BGM 付きの解説動画（横型 / 縦型 Shorts）を組み立てる Python パイプラインです。公開用に整理したサンプル実装です。

## 概要

スライド画像と台本から動画を作る工程を、再実行しやすい小さな段階に分けています。音声合成 → 図の部品分解とアニメ描画 → 字幕付きクリップ生成 → 連結 → BGM。各段階の成果物をファイルで残し、キャッシュのハッシュが一致すれば再計算しません。

## 主な機能・設計のポイント

- **段階実行**: `run_all.py` が synth → parallel_render → build → mix_bgm を順に実行。`--skip-*` で段階を飛ばし、`--continue-on-error` や `--qa-soft` で止め方を調整できます
- **ビルド前 QA（`qa_post_video.py`）**: 台本の構造的な問題を build 前に BLOCK / WARN で検出（字幕と読み上げ文の乖離、装飾記号の誤読、数字のかな化、共有スライドの扱い、解像度など）。事故を後段に持ち越さない設計です
- **冪等性・キャッシュ**: 音声は `sha256(テキスト+声+…)` をキーに再利用。アニメは `anim.meta.json` に入力 3 種のハッシュと尺を記録し、一致したら再レンダリングを省略します
- **字幕同期**: ElevenLabs の文字単位タイムスタンプ（`el_align.py`）を一次情報にし、`force_align.py`（faster-whisper）や `vad_align.py`（無音検出）を補助にします
- **合成音声の検証**: `verify_synth.py` が合成音声を Whisper で文字起こしして原文と突き合わせ、誤読の候補を機械的に抽出。読み辞書（kana_dict）に反映して再合成する運用です
- **図解アニメ**: 画像を連結成分（blob）に分け、並び順（order JSON）に沿って順番に出現させます。順序づけは LLM（`orderer_run.py`）、失敗時のルールベース代替（`blob_order_rulebased.py`）、監査と再試行（`audit_loop.py` / `audit_orders.py`）の三段構えで、諦めた場合は事実をレポートに残して build は止めません
- **字幕は成果物**: `compose_final.py` は字幕なしで作った動画に、全体時刻へ変換した字幕を最後に 1 回だけ焼き込みます
- **台本変更への追従**: `sync_slide_prompts.py` が台本のハッシュ差分を見て、画像生成プロンプトだけを再生成します

## ディレクトリ構成（主要ファイル）

```
slide-to-video-pipeline/
├ post_video/
│  ├ run_all.py            全段階の統合実行
│  ├ synth.py              ElevenLabs 合成（キャッシュ付き）
│  ├ parallel_render.py    図解アニメの並列レンダリング
│  ├ build.py / build_short.py   クリップ生成（横型 / 縦型）
│  ├ compose_final.py / mix_bgm.py / mix_bgm_short.py
│  ├ qa_post_video.py      ビルド前 QA
│  ├ el_align.py / force_align.py / vad_align.py / verify_synth.py
│  ├ blob_anim_prepare.py / orderer_run.py / audit_loop.py / order_merge.py ほか
│  └ paths.py / slides.py
├ audio_master.py  claude_cli.py  openai_image_config.py  _dotenv.py   ← 合成スタブ
```

## 動かし方

このコードは、ホストプロジェクトの `src/` 直下に置かれる前提で書かれています（`from src.post_video import …`、`paths.py` がリポジトリルートを 2 つ上と解釈、など）。そのまま `python run_all.py` では動かず、公開版は「設計と実装の閲覧」を主目的にしています。動かす場合は次を用意してください。

- Python 3、ffmpeg / ffprobe、下記「同梱していない依存」の Python パッケージ
- 環境変数: `ELEVENLABS_API_KEY`（音声合成）、`OPENAI_API_KEY`（スライド画像生成）、`POST_VIDEO_ROOT`（作業ルート。省略時はルート直下の `content_pipeline`）、`SUBTITLE_FONT_NAME`（ASS 字幕のフォント名。省略時は汎用の `sans-serif`）、`OVERLAY_FONT_PATH`（`generate_overlay.py` のラベル用フォントファイル。省略時は PIL の既定フォント）
- 声の ID は `slides.json` の `voice` または `--voice` で指定（`synth.py` の既定値はプレースホルダ）
- 実行例（`src` 配置後）: `python -m src.post_video.run_all --topic <topic> --workers 6`

## 制限事項（閉包）

`ast` と import 文の調査で、リポジトリ内・標準ライブラリに無く、`requirements.txt` も無いため、次を同梱していない依存として挙げます。

- **同梱していない Python パッケージ（9 件）**: `PIL`（Pillow）, `numpy`, `yaml`（PyYAML）, `soundfile`, `faster_whisper`, `whisperx`, `pykakasi`, `elevenlabs`, `openai`（一部は特定機能でのみ遅延 import）
- **他の公開リポジトリへの依存**: `diagram-blob-animator`（`diagram_blob.blob_renderer` / `auto_grouping`）。ビルド前 QA の単体テストは `video-qa-gates` 側にあります
- **パッケージ配置の前提**: `src`（`src.claude_cli`, `src.post_video`, `src.diagram_blob` を import するため、ホストの `src/` 配置が必要）
- **合成スタブ**: `audio_master.py`（音量調整は何もしない）, `claude_cli.py`（`run_llm_cli` のインターフェースのみ。実装は各自）, `openai_image_config.py`（定数のみ）, `_dotenv.py`（最小の .env ローダ）
- **非同梱の実行時データ**: 読み辞書 `tools/kana_dict.yaml`、BGM 素材、順序監査用のエージェント指示ファイル（`.claude/agents/order-auditor.md`、順序づけ担当の `.claude/agents/orderer.md`）、`slides.json` と画像
- 外部サービス: ElevenLabs、OpenAI 画像生成。音声・BGM・画像などの素材は、各サービスの利用規約と権利者の許諾の範囲で使ってください

## ライセンス

All rights reserved. 閲覧目的で公開しています。詳細は [LICENSE](LICENSE) を参照。
