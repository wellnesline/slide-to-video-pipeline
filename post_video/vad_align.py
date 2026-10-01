"""VAD (silence detection) ベースの caption entries → audio timing アラインメント。

force_align.align_entries_to_wav の char-level interpolation が破綻するケース
(per-entry ±1.6s 誤差 を観測) の代替として作成。

原理:
  - synth (ElevenLabs) は 句読点後に 250ms (。!?) / 120ms (、・) / 80ms (その他) の
    gap を挿入する (build.py:concat_slide_audio 仕様)
  - caption は 同じ punctuation で entry に分割される (split_caption_for_display)
  - 結果: line_wav 内の silence 数 ≈ caption entries 数 - 1
  - silence 開始位置 = 前 entry の終了 + 句読点
  - silence 終了位置 = 次 entry の発声開始

アルゴリズム:
  1. line_wav の RMS を 10ms 窓で計算
  2. RMS < threshold が gap_min_dur 以上 続く区間を silence として記録
  3. silence 数が entries 数 - 1 と一致するなら 各 entry に speech chunk を割り当て
  4. 一致しないなら None を返し caller に fallback (char-interpolation) を促す
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

# silence 検出パラメータ
RMS_WIN_SEC = 0.01       # 10ms 窓
SILENCE_RMS_TH = 0.015   # RMS < 0.015 を silence とみなす (経験値)
MIN_SILENCE_SEC = 0.06   # 60ms 以上 続く silence のみ count (gap 0.08-0.25 の最小値を カバー)


def _compute_rms_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """audio を 10ms 窓ごとに RMS。 envelope と 窓サイズ秒を返す。"""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    win = int(sr * RMS_WIN_SEC)
    if win <= 0:
        return np.array([], dtype=np.float32), 0.0
    n = len(audio) // win
    a = audio[:n*win].reshape(n, win)
    rms = np.sqrt(np.mean(a**2, axis=1))
    return rms, RMS_WIN_SEC


def _find_silences(rms: np.ndarray, win_sec: float,
                   threshold: float = SILENCE_RMS_TH,
                   min_dur_sec: float = MIN_SILENCE_SEC,
                   ) -> list[tuple[float, float]]:
    """RMS envelope から silence 区間 (start_sec, end_sec) を 列挙。"""
    silences = []
    in_sil = False
    sil_start_idx = 0
    for i, r in enumerate(rms):
        if r < threshold:
            if not in_sil:
                in_sil = True
                sil_start_idx = i
        else:
            if in_sil:
                dur = (i - sil_start_idx) * win_sec
                if dur >= min_dur_sec:
                    silences.append((sil_start_idx * win_sec, i * win_sec))
                in_sil = False
    # tail
    if in_sil:
        dur = (len(rms) - sil_start_idx) * win_sec
        if dur >= min_dur_sec:
            silences.append((sil_start_idx * win_sec, len(rms) * win_sec))
    return silences


PUNCT_CHARS = set("、。!?!?・")


def align_entries_by_silence(
    audio: np.ndarray,
    sr: int,
    entries: list[str],
    line_start_offset: float = 0.0,
) -> list[tuple[float, float]] | None:
    """line_wav の silence 検出で caption entries を audio chunk に対応付ける。

    各 entry 内に含まれる punctuation 数で「この entry が終わる silence index」 を 計算。
    audio 内の silence 数と 「entries 全体の punctuation 数」 を比較し、 一致すれば
    entry 境界を silence position に align。

    成功時: [(global_start, global_end)] * len(entries) を返す。
    失敗時 (silence 数 mismatch 等): None を返し caller に fallback を促す。
    """
    n_entries = len(entries)
    if n_entries <= 0:
        return []
    rms, win_sec = _compute_rms_envelope(audio, sr)
    if len(rms) == 0:
        return None
    silences = _find_silences(rms, win_sec)
    total_dur = len(audio) / sr
    # 末尾 silence (tail 無音) を除外
    silences = [s for s in silences if not (s[0] > total_dur * 0.95)]
    # 先頭 silence (head_pad) も除外
    silences = [s for s in silences if s[0] > 0.02]

    # 各 entry の cumulative punctuation count (= その entry が終わるまでの累積 punct 数)
    cum_punct = []
    cur = 0
    for e in entries:
        cur += sum(1 for c in e if c in PUNCT_CHARS)
        cum_punct.append(cur)

    if n_entries == 1:
        # 単 entry: 最初の音 onset 検出のみ
        first_sound_idx = 0
        for i, r in enumerate(rms):
            if r >= SILENCE_RMS_TH:
                first_sound_idx = i
                break
        return [(line_start_offset + first_sound_idx * win_sec, line_start_offset + total_dur)]

    # silence 数 vs 「最終 entry の cumulative punct」 が一致するか
    # (例: 4 entries, total 5 punct → 5 silences 期待。 最終 entry の cum_punct=5)
    # 厳密一致を期待するが、 最後の punct が tail 直前で silence として検出されない場合あり
    # (line_wav の末尾は 句読点+無音 だが silence 検出範囲外)。 ±1 まで許容。
    expected_silences = cum_punct[-1]
    if abs(len(silences) - expected_silences) > 1:
        return None

    # 最初の entry の開始: line 先頭から 最初の non-silence
    first_sound_idx = 0
    for i, r in enumerate(rms):
        if r >= SILENCE_RMS_TH:
            first_sound_idx = i
            break
    prev_end = first_sound_idx * win_sec

    timings: list[tuple[float, float]] = []
    for idx, _ in enumerate(entries):
        # この entry が「使う」 silence index = cum_punct[idx] - 1 (0-indexed)
        target_silence_idx = cum_punct[idx] - 1
        if target_silence_idx < len(silences):
            entry_end = silences[target_silence_idx][0]
            next_start = silences[target_silence_idx][1]
        else:
            entry_end = total_dur
            next_start = total_dur
        timings.append((line_start_offset + prev_end, line_start_offset + entry_end))
        prev_end = next_start
    return timings
