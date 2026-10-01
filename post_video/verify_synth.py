"""<post-id> 各 line の合成 wav を Whisper で文字起こし → 元 speech と diff 検出。

目的:
  ElevenLabs の誤読を機械検出 → kana_dict に追加すべき候補を抽出
  人間判定 → kana_dict 更新 → synth.py 再実行で 1 line 1 回だけ再合成

仕組みのループ:
  1. kana_dict で事前置換 → ElevenLabs 1 回目合成
  2. verify_synth.py で Whisper 検証
  3. mismatch 一覧から人間判定 → kana_dict 追加
  4. synth.py 再実行(辞書 update で cache_key 変わり、 該当 line だけ自動 cache miss → 1 回再合成)
  5. 同じ辞書のまま再 verify しても基本同じ結果 = 3 回目以降の課金は発生しない

Usage:
  python -m src.post_video.verify_synth --topic <post-id> [--workers 4] [--model medium]

出力:
  <topic>/work/audio/_verify_report.json   (mismatch 一覧)
  標準出力に diff 詳細
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import paths
from .slides import load_slides

# kana_dict を synth と同じロジックで適用(verify では「置換後 speech」を expected とする)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _worker(args):
    sid, li, wav_path, expected, model_size = args
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(wav_path), language="ja", vad_filter=True, beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()
        return (sid, li, expected, text, None)
    except Exception as e:
        return (sid, li, expected, "", str(e))


# 英字小辞書(略称 / 固有名詞をひらがな読みに、 longest first)
_ENG_YOMI = {
    'YouTube': 'ゆーちゅーぶ', 'youtube': 'ゆーちゅーぶ',
    'PDF': 'ぴーでぃーえふ', 'pdf': 'ぴーでぃーえふ',
    'API': 'えーぴーあい', 'api': 'えーぴーあい',
    'CSV': 'しーえすぶい', 'csv': 'しーえすぶい',
    'BGM': 'びーじーえむ', 'bgm': 'びーじーえむ',
    'YAML': 'やむる', 'yaml': 'やむる',
    'QA': 'きゅーえー', 'qa': 'きゅーえー',
    'OK': 'おーけー', 'ok': 'おーけー',
    'AI': 'えーあい', 'ai': 'えーあい',  # 短い英字は最後(誤マッチ防止)
}


def _kks():
    """pykakasi 1 回だけ初期化(プロセス毎、 import コスト回避)。"""
    if not hasattr(_kks, "_inst"):
        import pykakasi
        _kks._inst = pykakasi.kakasi()
    return _kks._inst


def _to_yomi(s: str) -> str:
    """ひらがな読みに正規化(漢字 → ひらがな + 英字略称ひらがな化 + カタカナ → ひらがな)。

    speech / Whisper 起こしの両方で同じ規則を適用 → 表記揺れを吸収。
    比較は編集距離 ratio で行う(_is_mismatch)。
    """
    # 1. 英字略称をひらがなに(longest first)
    for k in sorted(_ENG_YOMI.keys(), key=len, reverse=True):
        s = s.replace(k, _ENG_YOMI[k])
    # 2. 漢字 → ひらがな
    res = _kks().convert(s)
    yomi = ''.join(item['hira'] for item in res)
    # 3. カタカナ → ひらがな
    yomi = ''.join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in yomi)
    # 4. 記号 / 空白除去(数字は残す、 比較に有用)
    yomi = ''.join(c for c in yomi if c.isalnum() or 0x3040 <= ord(c) <= 0x309F)
    return yomi.lower()


_RATIO_THRESHOLD = 0.88  # SequenceMatcher.ratio() がこの値未満だと mismatch


def _is_mismatch(expected: str, actual: str) -> tuple[bool, float]:
    """ratio < threshold で mismatch。 (mismatch?, ratio) を返す。"""
    import difflib
    e = _to_yomi(expected)
    a = _to_yomi(actual)
    if not e or not a:
        return (e != a, 0.0)
    ratio = difflib.SequenceMatcher(None, e, a).ratio()
    return (ratio < _RATIO_THRESHOLD, ratio)


# 後方互換: 旧 _norm 名で呼ばれてもひらがな読みを返す
def _norm(s: str) -> str:
    return _to_yomi(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="medium",
                    choices=["tiny", "base", "small", "medium", "large-v3"])
    # build パイプ Step 1.5 自動 gate 化(sid_li bind bug 由来)
    # 旧来 verify_synth は単体実行で WARN だけ出していたが、 致命的誤読 (ratio < 0.5) が
    # build を通過する事故が連発したため、 BLOCK exit を導入。 デフォルトは strict OFF
    # (旧来動作維持)、 run_all/build_short が --strict を渡して BLOCK 判定する設計。
    ap.add_argument("--strict", action="store_true",
                    help="mismatch 率 > fatal-ratio-percent or 個別 ratio < fatal-line-threshold ≥ 1 件 で exit 1。 build パイプ Step 1.5 用。")
    ap.add_argument("--fatal-line-threshold", type=float, default=0.5,
                    help="この ratio 未満を「致命的 mismatch」 とみなす(默認 0.5)")
    ap.add_argument("--fatal-line-count", type=int, default=1,
                    help="致命的 mismatch ≥ N 件で BLOCK(默認 1)")
    ap.add_argument("--fatal-rate-percent", type=float, default=15.0,
                    help="mismatch 率 > N%% で BLOCK(默認 15.0%%)")
    args = ap.parse_args(argv)

    # synth と同じく kana_dict を適用した speech を expected にする
    from .synth import _load_kana_dict, _apply_kana_dict
    kana_dict = _load_kana_dict()

    slides = load_slides(args.topic)
    work_args = []
    for s in slides:
        sid = s["id"]
        for li, (line, wav) in enumerate(zip(s["lines"], s["audio_paths"])):
            if not wav.exists():
                continue
            expected = _apply_kana_dict(line["speech"], kana_dict)
            work_args.append((sid, li, wav, expected, args.model))

    print(f"=== verify_synth (topic={args.topic}, model={args.model}, workers={args.workers}) ===")
    print(f"  total lines: {len(work_args)}")
    print(f"  kana_dict keys: {len(kana_dict)}")
    print()

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, a) for a in work_args]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.time() - t0
    print(f"=== transcribe done in {elapsed:.1f}s ===\n")

    results.sort(key=lambda r: (r[0], r[1]))
    n_match = 0
    misreads: list[dict] = []
    for sid, li, expected, actual, err in results:
        if err:
            print(f"  [error] {sid}_{li:02d}: {err}")
            continue
        is_mis, ratio = _is_mismatch(expected, actual)
        if not is_mis:
            n_match += 1
        else:
            misreads.append({
                "sid": sid, "li": li, "expected": expected, "actual": actual,
                "ratio": round(ratio, 3),
            })

    print(f"=== {n_match} match / {len(misreads)} mismatch ({len(misreads)/len(results)*100:.0f}%) ===")
    print(f"=== threshold: ratio < {_RATIO_THRESHOLD} (ひらがな読み正規化 + difflib SequenceMatcher) ===\n")
    if misreads:
        print("=== mismatch 詳細(誤読候補、 ratio 昇順)===")
        for m in sorted(misreads, key=lambda x: x.get("ratio", 1.0)):
            print(f"\n--- slide_{m['sid']} L{m['li']:02d} (ratio={m.get('ratio', 0):.3f}) ---")
            print(f"  expected: {m['expected']}")
            print(f"  actual  : {m['actual']}")

    out = paths.audio_dir(args.topic) / "_verify_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(misreads, ensure_ascii=False, indent=2))
    print(f"\n→ saved: {out}")
    print(f"\n次:")
    print(f"  1. mismatch を人間レビュー → tools/kana_dict.yaml に追加")
    print(f"  2. python -m src.post_video.synth --topic {args.topic}(辞書 update で該当 line のみ自動再合成、 1 line 1 回まで)")

    # === --strict: BLOCK gate(build パイプ Step 1.5 自動化由来)===
    if args.strict and results:
        fatal = [m for m in misreads if m.get("ratio", 1.0) < args.fatal_line_threshold]
        mismatch_rate = len(misreads) / len(results) * 100
        if len(fatal) >= args.fatal_line_count:
            print(f"\n[verify_synth] BLOCK: ratio < {args.fatal_line_threshold} の致命的 mismatch が {len(fatal)} 件 (閾値 {args.fatal_line_count})")
            for m in fatal:
                print(f"  - slide_{m['sid']}_L{m['li']:02d}: ratio={m.get('ratio', 0):.3f}")
            print(f"  対処: 該当 line の speech を kana_dict 整備 or 表現変更 → synth 再合成 → 再 verify")
            return 1
        if mismatch_rate > args.fatal_rate_percent:
            print(f"\n[verify_synth] BLOCK: mismatch 率 {mismatch_rate:.1f}% > 閾値 {args.fatal_rate_percent}%")
            print(f"  対処: kana_dict 整備 or speech 全体見直し → synth 再合成 → 再 verify")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
