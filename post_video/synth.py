"""ElevenLabs v3 で slides.json の各 line を合成。

shared slide(shared_from)の line は既に shared プールに wav があるので、
topic/work/audio/ に copy するだけで合成スキップ。

post 個別 slide の line は ElevenLabs cache(cache_key = sha256(text+voice+...))を見て、
キャッシュヒットなら API 呼ばずに流用、未キャッシュなら API 合成。

Usage:
  python -m src.post_video.synth --topic <post-id> [--voice <VOICE_ID>]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths
from .slides import load_slides, load_top

# .env ロードは共通ローダー(_dotenv.py。リポジトリ直下)に集約。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dotenv import load_dotenv as _load_dotenv  # noqa: E402

# === kana_dict(全 ElevenLabs 系共通辞書)===
KANA_DICT_PATH = paths.repo_root() / "tools" / "kana_dict.yaml"


def _load_kana_dict() -> dict[str, str]:
    """tools/kana_dict.yaml をロード。 mappings 部分のみ返す。
    辞書は longest_first(長いキー優先)で適用。"""
    if not KANA_DICT_PATH.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(KANA_DICT_PATH.read_text(encoding="utf-8"))
    except ImportError:
        sys.stderr.write("[kana_dict] yaml モジュール不在、 辞書スキップ\n")
        return {}
    return data.get("mappings", {}) or {}


def _apply_kana_dict(text: str, mappings: dict[str, str]) -> str:
    """speech に辞書置換を適用。 長いキー優先で部分一致置換。"""
    if not mappings:
        return text
    # 長いキー優先(短いキーの誤マッチ回避)
    for key in sorted(mappings.keys(), key=len, reverse=True):
        text = text.replace(key, mappings[key])
    return text

VOICE_DEFAULT = "<VOICE_ID>"  # placeholder: set your own voice id (or pass --voice / slides.json "voice")
MODEL_ID = "eleven_v3"
SEED = 42
VOICE_SETTINGS = {
    "stability":         0.85,
    "similarity_boost":  0.85,
    "style":             0.0,
    "use_speaker_boost": True,
}
OUTPUT_FORMAT = "mp3_44100_192"

ROOT = paths.repo_root()
CACHE_DIR_BASE = ROOT / "cache/audio/elevenlabs"


def _cache_key(text: str, voice_id: str) -> str:
    payload = "|".join([
        text, voice_id, MODEL_ID, str(SEED),
        json.dumps(VOICE_SETTINGS, sort_keys=True),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _file_sha(p: Path) -> str:
    """ファイルの sha256 (16 文字 prefix)。 shared wav の content_hash 比較用。"""
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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


def force_delete_sid_caches(
    sid: str,
    slides: list[dict],
    audio_out: Path,
    cache_dir: Path,
    voice: str,
    kana_dict: dict[str, str],
) -> int:
    """sid 個別 slide の line wav + ElevenLabs cache_key mp3 を削除。

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
    for li, line in enumerate(target["lines"]):
        wav = audio_out / f"{sid}_{li:02d}.wav"
        if wav.exists():
            wav.unlink()
            n += 1
        # timestamps.json も削除 (mp3 と一緒に再生成必要)
        ts = audio_out / f"{sid}_{li:02d}.timestamps.json"
        if ts.exists():
            ts.unlink()
            n += 1
        speech = _apply_kana_dict(line["speech"], kana_dict)
        ck = _cache_key(speech, voice)
        mp3 = cache_dir / f"{ck}.mp3"
        if mp3.exists():
            mp3.unlink()
            n += 1
        cache_ts = cache_dir / f"{ck}.timestamps.json"
        if cache_ts.exists():
            cache_ts.unlink()
            n += 1
    return n


# === Line-level loudness normalization (hardlock) =====================
# 各 line wav を 必ず loudnorm 通して LUFS / true peak を統一する。
# 多話者(voice A / voice B など)で voice 別 / line 別の音量バラつき
# (-18.8 〜 -21.2 LUFS = 約 2.9 LUFS / 1.4 倍 のレンジ)を 物理的に解消する。
# 仕様変更は change-validator 経由でのみ可、 yaml/CLI からの override 不可。
# 値の根拠: I=-18 は post 系で実績ある中庸ターゲット、 LRA=11 は会話自然、
#          TP=-2 は alimiter とのヘッドルーム確保。
LOUDNORM_I = -18.0          # Integrated loudness target (LUFS)
LOUDNORM_LRA = 11.0         # Loudness range target
LOUDNORM_TP = -2.0          # True peak max (dBTP)

# === Tail breath/click hardlock  =====================
# ElevenLabs v3 の文末に入る「すっ」「ふっ」等の息継ぎ/吸気音は silenceremove
# (-32dB threshold)を超えるため(検出例: max -2.1dB / mean -15dB)、 threshold
# 緩和では取り切れない。 一律 150ms 末尾カットで物理排除。
# 正規音声の最後 150ms が削れるが、 voice 学習データ的に最後音節は明瞭で 違和感ほぼ無し。
TAIL_TRIM_MS = 150


def _mp3_to_wav(mp3: Path, wav: Path) -> None:
    """mp3 → wav 変換時に 末尾 trail trim + 2 pass loudnorm で 厳密 LUFS 統一(hardlock)。

    2 pass loudnorm の理由:
      1 pass mode では 短い line (1-3秒) で 動的補正が target に届かず、 voice/line 別の
      LUFS バラつきが 残る(1 pass では改善が見られなかった)。
      2 pass で 1st 計測 → linear=true で 厳密 normalize する。

    事故由来:
      - 文末に変なノイズが乗る事象 → -32dB トリム導入(50ms keep)
      - 末尾の保持を 50ms → 150ms → 200ms と段階的に緩和(語尾が切れて聞こえるため)
      - 多話者(話者A + 別 voice)で line 別 LUFS バラつき → 2 pass loudnorm hardlock
    全 channel / 全 <post-id> 系で同症状の根治。"""
    import json as _json
    import re as _re
    trail_filter = "areverse,silenceremove=start_periods=1:start_silence=0.20:start_threshold=-32dB,areverse"
    # 末尾 TAIL_TRIM_MS ms 一律カット用 duration 計算
    dur_str = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)
    ]).decode().strip()
    src_dur = float(dur_str)
    trim_t = max(0.1, src_dur - TAIL_TRIM_MS / 1000.0)
    # === タイムスタンプ駆動の語尾保護(利用者「文末が話し終える前に切れる」) ===
    # 固定 TAIL_TRIM_MS の一律末尾カットは、 語尾が長く伸びる voice で発話本体を食う。
    # ElevenLabs char-level timestamps の最後の文字 ends を ground truth として、
    # 「最後の発話文字の終わり」 より手前では絶対に切らない(息継ぎ音は ends より後ろなので
    # silenceremove + ends 以降の余白で処理される)。 timestamps 不在 line は従来挙動のまま。
    try:
        _ts_p = mp3.with_suffix(".timestamps.json")
        if _ts_p.exists():
            _ts = _json.loads(_ts_p.read_text(encoding="utf-8"))
            _ends = _ts.get("normalized_ends") or _ts.get("ends") or []
            if _ends:
                _speech_end = float(_ends[-1])
                # 語尾の自然な減衰のため ends + 30ms を下限に(息継ぎ音より十分手前)
                trim_t = max(trim_t, min(src_dur, _speech_end + 0.03))
    except Exception:
        pass  # timestamps 読込失敗時は従来の固定カットにフォールバック
    # === Pass 1: 入力の loudness 計測 ===
    measure_filter = (
        f"{trail_filter},"
        f"loudnorm=I={LOUDNORM_I}:LRA={LOUDNORM_LRA}:TP={LOUDNORM_TP}:print_format=json"
    )
    p1 = subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3), "-af", measure_filter, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # ffmpeg は stderr に loudnorm JSON を吐く
    m = _re.search(r"\{[^{}]+\}", p1.stderr or "", _re.DOTALL)
    if not m:
        # measurement 失敗時は 1 pass fallback(極端な短素材等)
        cmd = [
            "ffmpeg", "-y", "-i", str(mp3),
            "-ar", "44100", "-ac", "1",
            "-af", f"{trail_filter},loudnorm=I={LOUDNORM_I}:LRA={LOUDNORM_LRA}:TP={LOUDNORM_TP}",
            "-t", f"{trim_t:.3f}",
            str(wav),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return
    measured = _json.loads(m.group(0))
    # === Pass 2: 厳密 normalize ===
    linear_filter = (
        f"{trail_filter},"
        f"loudnorm=I={LOUDNORM_I}:LRA={LOUDNORM_LRA}:TP={LOUDNORM_TP}:"
        f"measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true:print_format=summary"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3),
        "-ar", "44100", "-ac", "1",
        "-af", linear_filter,
        "-t", f"{trim_t:.3f}",
        str(wav),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


_CURRENT_VOICE = VOICE_DEFAULT  # _synth_one 用 globals(main で上書き)


def _synth_one(client, text: str, mp3_out: Path, voice_id: str | None = None) -> int:
    """ElevenLabs convert_with_timestamps で mp3 + char-level timestamps を 同時取得。
    timestamps は mp3 の隣に <stem>.timestamps.json として 保存。
    字幕同期 完璧化のため convert → convert_with_timestamps に切替。
    ASR (whisper/whisperx) は fallback chain の 2-3 段目に降格、 通常は TTS engine の
    ground truth char timing を そのまま 字幕配置に使う (ASR で ±30〜1600ms の字幕ずれが出ていた問題への対策)。
    """
    import base64 as _b64
    # 多話者対応: voice_id 引数 > _CURRENT_VOICE
    vid = voice_id or _CURRENT_VOICE
    resp = client.text_to_speech.convert_with_timestamps(
        voice_id=vid,
        model_id=MODEL_ID,
        text=text,
        output_format=OUTPUT_FORMAT,
        voice_settings=VOICE_SETTINGS,
        seed=SEED,
    )
    mp3_bytes = _b64.b64decode(resp.audio_base_64)
    mp3_out.write_bytes(mp3_bytes)
    # timestamps を JSON で保存 (mp3 の sibling)
    ts_out = mp3_out.with_suffix(".timestamps.json")
    align = resp.alignment
    normalized = resp.normalized_alignment
    ts_data = {
        "schema": "el_char_align_v1",
        "engine": "elevenlabs",
        "model_id": MODEL_ID,
        "voice_id": vid,
        "seed": SEED,
        "text": text,  # kana_dict 適用後の speech
        "characters": list(align.characters),
        "starts": list(align.character_start_times_seconds),
        "ends": list(align.character_end_times_seconds),
        "normalized_characters": (list(normalized.characters) if normalized else None),
        "normalized_starts": (list(normalized.character_start_times_seconds) if normalized else None),
        "normalized_ends": (list(normalized.character_end_times_seconds) if normalized else None),
    }
    ts_out.write_text(json.dumps(ts_data, ensure_ascii=False, indent=None), encoding="utf-8")
    return len(mp3_bytes)


def main(argv: list[str] | None = None) -> int:
    global _CURRENT_VOICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--voice", default=None,
                    help="ElevenLabs voice_id。 未指定なら slides.json の voice、 さらに無ければ VOICE_DEFAULT")
    ap.add_argument("--workers", type=int, default=4,
                    help="ElevenLabs API 並列呼び出し数(default 4)")
    ap.add_argument("--force-slides", default=None,
                    help="カンマ区切り sid。 指定 slide の wav + ElevenLabs cache_key mp3 を事前削除してから synth。 "
                         "例: --force-slides 03,07 (shared セクションは保護)")
    ap.add_argument("--only-slides", default=None,
                    help="カンマ区切り sid。 指定 slide のみ処理、 他 sid は完全 skip。 "
                         "UI の per-scene 再生成ボタン用、 --force-slides と併用で アトミック再生成")
    ap.add_argument("--self-heal", action="store_true",
                    help="synth 後 verify_synth(Whisper)で誤読検出 → 該当 line の cache 削除 → 再合成 1 回 → 再 verify")
    ap.add_argument("--self-heal-model", default="medium",
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    help="--self-heal で使う Whisper モデル(default medium)")
    args = ap.parse_args(argv)

    topic = args.topic
    # voice 優先順位: --voice CLI > slides.json top.voice > VOICE_DEFAULT
    # (過去事故由来: slides.json に voice 書いたのに反映されなかった)
    if args.voice:
        voice = args.voice
        voice_source = "cli arg"
    else:
        top = load_top(topic)
        voice = top.get("voice") or VOICE_DEFAULT
        voice_source = "slides.json" if top.get("voice") else "default"
    _CURRENT_VOICE = voice
    print(f"[voice] {voice} (source: {voice_source})")
    _load_dotenv()

    cache_dir = CACHE_DIR_BASE / voice / "v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_out = paths.audio_dir(topic)
    audio_out.mkdir(parents=True, exist_ok=True)

    slides = load_slides(topic)

    # === kana_dict ロード(全 ElevenLabs 系共通辞書)===
    kana_dict = _load_kana_dict()
    if kana_dict:
        print(f"[kana_dict] {KANA_DICT_PATH.relative_to(paths.repo_root())} loaded ({len(kana_dict)} keys)")

    # === --force-slides: 指定 slide の cache を事前削除(post 個別のみ、 shared は保護)===
    # synth.py には --force 全体フラグが無いので優先順位判定不要(他 2 CLI と異なる)。
    force_slides = _parse_force_slides(args.force_slides)
    if force_slides:
        print(f"[--force-slides] {sorted(force_slides)} の cache を削除")
        deleted_total = 0
        for sid in sorted(force_slides):
            deleted_total += force_delete_sid_caches(sid, slides, audio_out, cache_dir, voice, kana_dict)
        print(f"  → {deleted_total} 件削除")

    # === --only-slides: 指定 sid 以外を完全 skip ===
    only_slides = _parse_force_slides(args.only_slides)
    if only_slides:
        before = len(slides)
        slides = [s for s in slides if s["id"] in only_slides]
        print(f"[--only-slides] {sorted(only_slides)} のみ処理 ({before} → {len(slides)} slides)")
        if not slides:
            print("  (該当 slide 無し、 何もせず終了)")
            return 0

    # === Preflight ===
    # 多話者対応: plan の 8 番目に line_voice 追加
    plan: list[tuple] = []  # (sid, li, speech, mp3_path, wav_target, cached, shared, line_voice)
    for s in slides:
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            # speech に kana_dict 置換を適用してから cache_key 計算
            raw_speech = line["speech"]
            speech = _apply_kana_dict(raw_speech, kana_dict)
            wav_target = audio_out / f"{sid}_{li:02d}.wav"
            # line.voice (compose で speaker 解決済) > top voice
            line_voice = line.get("voice") or voice
            if s["shared_id"]:
                # shared: source wav は shared から借りる
                src_wav = s["audio_paths"][li]
                plan.append((sid, li, speech, None, wav_target, True, src_wav, line_voice))
            else:
                # line voice ごとに cache_dir を分ける(voice 別 cache 隔離)
                line_cache_dir = CACHE_DIR_BASE / line_voice / "v3"
                line_cache_dir.mkdir(parents=True, exist_ok=True)
                ck = _cache_key(speech, line_voice)
                mp3_p = line_cache_dir / f"{ck}.mp3"
                # mp3 + timestamps.json 両方揃って 初めて「cached」 扱い
                # (timestamps.json 不在 = 旧 convert() で 生成された mp3 = 再 synth 必要)
                ts_p = line_cache_dir / f"{ck}.timestamps.json"
                cached = mp3_p.exists() and ts_p.exists()
                plan.append((sid, li, speech, mp3_p, wav_target, cached, None, line_voice))

    n_total = len(plan)
    # plan の 7番目 (index 6) は shared_src、 多話者で 8番目に line_voice 追加
    n_shared = sum(1 for p in plan if p[6] is not None)
    n_cached = sum(1 for p in plan if p[5] and p[6] is None)
    # voice 別 line 数を表示
    voice_counts: dict[str, int] = {}
    for p in plan:
        lv = p[7]
        voice_counts[lv] = voice_counts.get(lv, 0) + 1
    n_uncached = n_total - n_shared - n_cached
    chars_uncached = sum(len(p[2]) for p in plan if not p[5])

    print(f"=== Preflight (topic={topic}, voice={voice}) ===")
    print(f"  Total lines: {n_total}")
    print(f"  Shared:      {n_shared} (shared プールから流用)")
    print(f"  Cached:      {n_cached} (ElevenLabs API cache hit)")
    print(f"  Uncached:    {n_uncached} ({chars_uncached} chars, ${chars_uncached * 2 * 99 / 500000:.2f} 概算)")
    if len(voice_counts) > 1:
        print(f"  voices:      {voice_counts} (多話者構成)")
    print()

    # === Synthesize(ThreadPoolExecutor 並列、 default 4 workers)===
    if n_uncached > 0:
        from elevenlabs.client import ElevenLabs
        from concurrent.futures import ThreadPoolExecutor, as_completed
        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        # 多話者対応: line_voice を targets に含めて _synth_one に渡す
        targets = [(sid, li, speech, mp3_p, line_voice) for sid, li, speech, mp3_p, _, cached, shared_src, line_voice in plan
                   if not cached and shared_src is None]
        n_fail = 0

        def _worker(args):
            sid, li, speech, mp3_p, line_voice = args
            try:
                size = _synth_one(client, speech, mp3_p, voice_id=line_voice)
                return f"  [synth] {sid}_{li:02d} ({len(speech)}字, {size:,}B, v={line_voice[:8]}) | {speech[:30]}"
            except Exception as e:
                return f"  [error] {sid}_{li:02d}: {e}"

        print(f"  parallel synth: {len(targets)} lines × {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(_worker, t) for t in targets]):
                msg = f.result()
                print(msg, flush=True)
                if msg.startswith("  [error]"):
                    n_fail += 1
        if n_fail > 0:
            return 1

    # === wav 出力(post 個別: mp3→wav 変換、shared: copy)===
    print(f"\n=== wav out → {audio_out} ===")
    for sid, li, speech, mp3_p, wav_t, cached, shared_src, line_voice in plan:
        if shared_src is not None:
            # shared wav copy: content_hash 比較で再 copy 判定
            # 旧来は mtime 比較だったが、shutil.copy2 が mtime を保持するため一度 copy 後に
            # shared 側が再合成されても「同じ mtime」 で再 copy が trigger されない
            # 構造的バグ(parallel_render の同型対策と同時 監査由来)。
            need_copy = not wav_t.exists()
            if not need_copy and _file_sha(shared_src) != _file_sha(wav_t):
                need_copy = True
            if need_copy:
                shutil.copy2(shared_src, wav_t)
            # shared プールに timestamps.json があれば 一緒に copy
            shared_ts = shared_src.with_suffix(".timestamps.json")
            if shared_ts.exists():
                shutil.copy2(shared_ts, wav_t.with_suffix(".timestamps.json"))
        else:
            if not mp3_p.exists():
                print(f"  [skip] {sid}_{li:02d} (mp3 missing)")
                continue
            # NOTE: 旧来は `wav_t.mtime > mp3_p.mtime` で skip していたが、
            # line の text 並び替え時に「mp3 cache hit + 旧 wav 残存」 で
            # 旧 line の音声が新 slide に紐付く事故が連発(5 line 影響、 ratio < 0.2)。
            # mp3 → wav 変換は 1 秒未満なので、 常に上書き再変換することで wav と
            # 現 line の text を必ず同期させる(audio_master 系の trailing trim も常に適用される副次効果)。
            try:
                _mp3_to_wav(mp3_p, wav_t)
                # cache に timestamps.json があれば audio_dir に copy
                mp3_ts = mp3_p.with_suffix(".timestamps.json")
                if mp3_ts.exists():
                    shutil.copy2(mp3_ts, wav_t.with_suffix(".timestamps.json"))
            except Exception as e:
                print(f"  [conv error] {sid}_{li:02d}: {e}")

    # === --self-heal: verify → re-synth mismatched → re-verify===
    # ElevenLabs 誤読を半自動修正。 cache 削除 → 同じ speech で 1 回再合成
    # (運用方針: 誤読時は同じテキストで再合成するのが第一手)。
    # 1 line あたり最大 1 回限り(コスト爆増防止)、 残った誤読は WARN で人間 kana_dict 介入要請。
    if args.self_heal:
        print("\n=== self-heal: verify → re-synth mismatched lines (1 pass) ===")
        py_exe = sys.executable
        rc = subprocess.run(
            [py_exe, "-m", "src.post_video.verify_synth",
             "--topic", topic, "--workers", str(args.workers),
             "--model", args.self_heal_model],
            cwd=str(paths.repo_root()),
        ).returncode
        report_path = paths.audio_dir(topic) / "_verify_report.json"
        if not report_path.exists():
            print("  [self-heal] verify report 不在、 終了")
        else:
            misreads = json.loads(report_path.read_text(encoding="utf-8"))
            if not misreads:
                print("  [self-heal] mismatch 0 件、 self-heal 不要")
            else:
                print(f"  [self-heal] {len(misreads)} 件 mismatch → 該当 line の cache 削除")
                for m in misreads:
                    sid, li = m["sid"], m["li"]
                    speech = m["expected"]  # verify_synth は kana_dict 適用後を expected に格納
                    ck = _cache_key(speech, voice)
                    cache_mp3 = cache_dir / f"{ck}.mp3"
                    wav = audio_out / f"{sid}_{li:02d}.wav"
                    if cache_mp3.exists():
                        cache_mp3.unlink()
                    if wav.exists():
                        wav.unlink()
                print(f"  [self-heal] re-synth({len(misreads)} lines)...")
                rc2 = subprocess.run(
                    [py_exe, "-m", "src.post_video.synth",
                     "--topic", topic, "--voice", voice, "--workers", str(args.workers)],
                    cwd=str(paths.repo_root()),
                ).returncode
                if rc2 != 0:
                    print("  [self-heal] re-synth 失敗、 self-heal 中断")
                else:
                    print(f"  [self-heal] re-verify...")
                    subprocess.run(
                        [py_exe, "-m", "src.post_video.verify_synth",
                         "--topic", topic, "--workers", str(args.workers),
                         "--model", args.self_heal_model],
                        cwd=str(paths.repo_root()),
                    )
                    misreads2 = []
                    if report_path.exists():
                        misreads2 = json.loads(report_path.read_text(encoding="utf-8"))
                    print(f"  === self-heal done: {len(misreads)} → {len(misreads2)} mismatch remaining ===")
                    if misreads2:
                        print(f"  [self-heal] WARN: 残存 mismatch {len(misreads2)} 件 → kana_dict.yaml に追加して synth 再実行を検討")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
