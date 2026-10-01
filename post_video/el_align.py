"""ElevenLabs convert_with_timestamps の char-level timing を字幕 entries に紐付ける。

新設。 <post-id> 字幕同期 を ASR 推定 (±10-1600ms 誤差) から
TTS engine の ground truth (±1-5ms) に切替える 1 次ソース。

入力:
  - entries_kana: caption を split + kana_dict 適用後の文字列リスト
  - ts: timestamps.json をパースした dict (schema=el_char_align_v1)
  - line_start_offset: concat_wav 内の本 line の speech 開始位置 (= line["start"] + HEAD_PAD)
  - wav_dur (optional): trailing trim 後の wav 実長。 timestamps end が これを超えたら clamp

出力:
  - [(entry_start_sec, entry_end_sec)] * len(entries) を ASS-clip-local 時刻で返す
  - 失敗時 None を返し caller に fallback (whisperx 等) を促す
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = "el_char_align_v1"


def load_timestamps(audio_dir: Path, sid: str, li: int) -> Optional[dict]:
    """audio_dir/<sid>_<li>.timestamps.json を読み込む。 不在 / parse 失敗 → None。"""
    p = audio_dir / f"{sid}_{li:02d}.timestamps.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [el_align] {p.name} parse failed: {e}", file=sys.stderr)
        return None
    # schema 検証
    if d.get("schema") != SCHEMA_VERSION:
        print(f"  [el_align] {p.name} unknown schema {d.get('schema')!r}", file=sys.stderr)
        return None
    chars = d.get("characters") or []
    starts = d.get("starts") or []
    ends = d.get("ends") or []
    if not (len(chars) == len(starts) == len(ends)):
        print(f"  [el_align] {p.name} chars/starts/ends length mismatch", file=sys.stderr)
        return None
    return d


def _char_index_map(entries_kana: list[str], ts_text: str) -> Optional[list[int]]:
    """entries_kana の concat と timestamps の text を difflib で 1:1 char マッチ。

    Returns:
      cap_idx → ts_char_idx の list (len = sum(entries_kana の長さ))
      マッチ率が 70% 未満なら None (caller fallback)
    """
    full_cap = "".join(entries_kana)
    if not full_cap or not ts_text:
        return None
    sm = difflib.SequenceMatcher(None, full_cap, ts_text)
    matching = sm.get_matching_blocks()
    cap_to_ts: dict[int, int] = {}
    matched_chars = 0
    for m in matching:
        matched_chars += m.size
        for k in range(m.size):
            cap_to_ts[m.a + k] = m.b + k
    match_ratio = matched_chars / max(len(full_cap), 1)
    if match_ratio < 0.70:
        return None
    # caption char に対応する ts char idx を 線形補間
    sorted_keys = sorted(cap_to_ts.keys())
    out: list[int] = []
    for i in range(len(full_cap)):
        if i in cap_to_ts:
            out.append(cap_to_ts[i])
            continue
        # 補間: 前後の matched key から
        prev_k = None
        next_k = None
        for k in sorted_keys:
            if k < i:
                prev_k = k
            elif k > i:
                next_k = k
                break
        if prev_k is None and next_k is None:
            out.append(0)
        elif prev_k is None:
            out.append(cap_to_ts[next_k])
        elif next_k is None:
            out.append(cap_to_ts[prev_k])
        else:
            # 線形補間で 推定 char index
            ratio = (i - prev_k) / max(next_k - prev_k, 1)
            est = cap_to_ts[prev_k] + (cap_to_ts[next_k] - cap_to_ts[prev_k]) * ratio
            out.append(int(round(est)))
    return out


def align_entries_from_timestamps(
    entries_kana: list[str],
    entries_display: list[str],
    ts: dict,
    line_start_offset: float,
    wav_dur: Optional[float] = None,
) -> Optional[list[tuple[float, float]]]:
    """timestamps.json を ground truth として entries → (start, end) を算出。

    Args:
      entries_kana: kana_dict 適用後の entries (timestamps の characters と char 整合)
      entries_display: 表示用 entries (戻り値の対応用、 ASS には こちらが書かれる)
      ts: load_timestamps() の返り値
      line_start_offset: concat_wav 内 speech 開始位置 (= line["start"] + HEAD_PAD)
      wav_dur: trailing trim 後の wav 実長 (timestamps の end clamp 用)
    """
    if len(entries_kana) != len(entries_display):
        return None
    # normalized_alignment があれば それを優先 (apply_text_normalization=auto 時)
    chars = ts.get("normalized_characters") or ts.get("characters") or []
    starts = ts.get("normalized_starts") or ts.get("starts") or []
    ends = ts.get("normalized_ends") or ts.get("ends") or []
    if not (chars and starts and ends and len(chars) == len(starts) == len(ends)):
        return None

    ts_text = "".join(chars)
    cap_idx_to_ts = _char_index_map(entries_kana, ts_text)
    if cap_idx_to_ts is None:
        return None

    timings: list[tuple[float, float]] = []
    cum_cap_idx = 0
    for ek in entries_kana:
        st_cap = cum_cap_idx
        et_cap = cum_cap_idx + len(ek) - 1
        if st_cap >= len(cap_idx_to_ts) or et_cap >= len(cap_idx_to_ts):
            return None
        st_ts_idx = cap_idx_to_ts[st_cap]
        et_ts_idx = cap_idx_to_ts[et_cap]
        # clamp idx 範囲
        st_ts_idx = max(0, min(st_ts_idx, len(starts) - 1))
        et_ts_idx = max(0, min(et_ts_idx, len(ends) - 1))
        entry_start = float(starts[st_ts_idx])
        entry_end = float(ends[et_ts_idx])
        # trim clamp (wav_dur 超え 防止)
        if wav_dur is not None:
            entry_start = min(entry_start, wav_dur)
            entry_end = min(entry_end, wav_dur)
        # 単調増加保証
        if entry_end < entry_start:
            entry_end = entry_start
        # 全体 offset 加算 (concat_wav 上の 絶対時刻)
        timings.append((line_start_offset + entry_start, line_start_offset + entry_end))
        cum_cap_idx += len(ek)

    # 単調増加 + 重畳禁止 (隣接 entry の overlap 解消、 既存 build.py の sanity と整合)
    for i in range(len(timings) - 1):
        st_i, et_i = timings[i]
        st_next = timings[i + 1][0]
        target_end = st_next - 0.01
        if et_i > target_end and target_end > st_i:
            timings[i] = (st_i, target_end)

    return timings
