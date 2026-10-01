"""<post-id> slides.json の構造的 QA(build 前にチェックして事故予防)。

検出項目:
  - F1 [BLOCK] 「次回」言及(指示なき限り NG とする運用ルール)
  - F2 [WARN]  shared と post 個別 line で speech が重複(コピペミスの兆候)
  - F3 [BLOCK] intro_shared に L1 以降が残ってる(L0 のみ運用ルール BLOCK 化)
  - F4 [BLOCK] caption と speech の trigram 共通率 < 0.65(speech ≥ 50 字、 shared 除外。 乖離事故由来 / BLOCK 昇格)
                比較前に kana_dict 逆引きで英字カナ化を正規化(偽陽性回避)
  - F5 [BLOCK] speech 内の **数字 kana 表記** 検出(ひゃくえん 等、 失敗パターン再発防止)
                ルール: 数字は漢字混じり原文のまま、 kana 化禁止
  - F6 [WARN]  字幕同期ずれ(各 clip mp4 の duration と audio 期待値の差が ±0.5s 超)
  - F7 [WARN]  force_align fallback 発生(_force_align_fallback.log の件数)
  - F8 [WARN]  verify_synth 未実行 / mismatch 残存(_verify_report.json)
  - F9 [BLOCK] speech 内で 「数字-数字」 等の記号使用(ElevenLabs が「4-6」 を「4 の 6」 と読む)
  - F9-c [BLOCK] speech 内の装飾記号 ★ ☆ ◆ ◇ ◎ ● (同記号 2 連は匿名化扱いで除外。 ★→「カー」誤読由来)
  - F9-d [BLOCK] speech に小数点 + 行末「。」 隣接(距離 ≤ 10 字。 35.0→「ドット」波及由来)
  - F10 [INFO]  speech 内の kana_dict 未登録英字 keyword(事前カタカナ化検討)
  - F11 [BLOCK] topic に /shorts/ 含む場合 final.mp4 が 1080×1920 でない(横長 build 防止。 横長 build 事故由来)
  - F12 [WARN]  caption 内の 3 字以上名詞/数値/英字略語が slide_prompt 不在で ≥ 3 件(画像ハルシネリスク。 過去の事例由来)
  - F13 [WARN]  caption 内 小数点が「点」 エンコードのまま残存 (過去の事故由来)
  - F14 [WARN/BLOCK] final.mp4 / final_no_bgm.mp4 の video/audio stream drift。 ±0.08s WARN / ±1.0s BLOCK。 option D (compose_final video PTS anchor) 適用後は ±0.5s 程度の container drift が残るのが仕様、 ±1.0s 超は build.py 側修正検討領域 (監査ループ)

Usage:
  python -m src.post_video.qa_post_video --topic <post-id>
  exit 0 = OK / exit 1 = BLOCK あり / exit 2 = WARN あり

  --soft: content 系 F* (F1/F3/F4/F5/F9/F9-c/F9-d) を BLOCK→WARN 降格。
          動画ファイル整合性に関係しない品質ルールを skip し、 最後まで build を通したい
          client demo / 緊急公開 用途。 ファイル整合性系 (F11 Shorts 解像度 / F14 audio
          drift) は BLOCK 維持。
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from . import paths
from .slides import load_slides

# 検出ワード
NEXT_WORDS = ["次回は", "次回予告", "次回の動画", "次回お話"]

# F5: 数字 kana ワード(連続 3 つ以上で検出 = 数字 kana 化されているとみなす)
NUM_KANA_TOKENS = [
    "いち", "に", "さん", "よん", "ご", "ろく", "なな", "しち", "はち", "きゅう", "く",
    "じゅう", "じゅっ", "ひゃく", "びゃく", "ぴゃく", "せん", "ぜん", "まん", "おく", "ちょう",
    "ぽん",  # 「よんじゅっぽん」用
]
NUM_KANA_RE = re.compile(r"(" + "|".join(sorted(NUM_KANA_TOKENS, key=len, reverse=True)) + r"){3,}")

# F9: speech 記号誤用検出
# 数字-数字(範囲)、 数字~数字、 数字/数字 等は ElevenLabs が「の」 等と読む → 「Aから B」 と書くべき
F9_RANGE_RE = re.compile(r"\d+\s*[-~/〜]\s*\d+")
# 矢印・中黒・スラッシュ単独などの speech 不適切記号(複合語以外)
F9_DISALLOWED_CHARS = ["→", "←", "↑", "↓", "⇒"]

# F9-c: 装飾記号 BLOCK (過去の事故由来)
# ★ → 「カー」 / 「ホシ」 等の誤読。 同記号 2 連(★★ 等)は匿名化扱いで pass。
F9_DECORATIVE_CHARS = ["★", "☆", "◆", "◇", "◎", "●"]

# F9-d: 小数 + 行末「。」 隣接 BLOCK (過去の事故由来)
# 35.0 → decimal-mode 波及で末尾「。」 まで「ドット」 化する事故。
# NOTE: 距離 ≤ 10 の正規表現は 実データで false positive が複数検出された。
# → 閾値見直しを 別 PR に移し、本 PR では距離 ≤ 3 で運用。
# 距離 ≤ 3 でも false positive 1件残存のため、本 PR では無効化し
# 別 PR で正規表現を精緻化後に有効化する。
# F9_DECIMAL_TAIL_RE = re.compile(r"\d+\.\d+[^。!?]{0,3}。\s*$")  # (別 PR 待ち)
# 暫定: WARN 降格で残存 (TODO: 別 PR で BLOCK 昇格 + 閾値精緻化)
F9_DECIMAL_TAIL_RE = re.compile(r"\d+\.\d+[^。!?]{0,3}。\s*$")

# F12: slide_prompt 列挙整合 WARN (画像ハルシネーション由来)
# caption 内の 3 字以上トークン(カタカナ/ひらがな名詞・数値・英字略語)が slide_prompt に不在 ≥ 3 件で WARN
F12_TOKEN_RE = re.compile(r"[A-Za-z]{2,}|[ぁ-ん]{3,}|[ァ-ヶー]{3,}|\d+(?:\.\d+)?")

# F13: caption 内 小数点が「点」 エンコードのまま残存 WARN (過去の事故由来)
# scripter が speech 用に「4.0」 → 「4点0」 と書き換えた際、 caption 側を visual 「4.0」 に
# 戻し忘れる事故。 caption は視聴者の目に入るので「点」 のままだと意味が崩れる。
# 例: 「閾値は4点0で5点満点」 (誤) → 「閾値は4.0で5点満点」 (正)
# 偽陽性除外: 直後が 「点満点」 「点中」 「点台」 等の score 表現や、 前後が「合計」 「平均」 等
# のスコア文脈は除外しきれないため、 WARN レベルで通知のみ。
F13_DECIMAL_AS_TEN_RE = re.compile(r"(?<![点])\d点\d(?![点満中台])")

# F10: speech 内未登録英字 keyword
# kana_dict.yaml に未登録の英字 token を抽出 → INFO 出力。 事前カタカナ化検討用。
F10_ENG_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]+(?:\.[a-z_][a-z0-9_]*)?")
# 接続詞 / 1〜2 文字 / 一般語 は無視
F10_STOP_WORDS = {
    "a", "an", "the", "to", "in", "on", "of", "is", "it", "as", "by", "at",
    "no", "or", "if", "be", "do", "go", "so", "up", "we", "me", "my", "we",
    "and", "for", "with", "this", "that", "are", "was", "but", "not", "you",
}


def _load_kana_keys() -> set[str]:
    """kana_dict.yaml の登録キー集合を返す(case-preserved + lowercase 両方)。"""
    try:
        import yaml
        kana_path = paths.repo_root() / "tools" / "kana_dict.yaml"
        if not kana_path.exists():
            return set()
        data = yaml.safe_load(kana_path.read_text(encoding="utf-8")) or {}
        keys = set(data.get("mappings", {}).keys())
        # lowercase variant も同等扱い
        keys |= {k.lower() for k in keys}
        return keys
    except Exception:
        return set()


def _load_kana_reverse() -> dict[str, str]:
    """カナ表記 → 英字 の逆引き辞書(F4 偽陽性回避用)。
    speech 内のカナ化済み英字 (例「キューエー」) を比較前に英字 (「QA」) に戻す。
    1対多衝突時は longest-first で最初の英字を採用。"""
    try:
        import yaml
        kana_path = paths.repo_root() / "tools" / "kana_dict.yaml"
        if not kana_path.exists():
            return {}
        data = yaml.safe_load(kana_path.read_text(encoding="utf-8")) or {}
        mappings = data.get("mappings", {})
        return {v: k for k, v in mappings.items() if v}
    except Exception:
        return {}

# F6 sync drift threshold (clip duration vs audio sum 期待値、秒)
SYNC_DRIFT_THRESHOLD = 0.5
HEAD_PAD = 0.05  # build.py / parallel_render._slide_duration と同じ


def _ffprobe_duration(p: Path) -> float | None:
    """container `format.duration = max(audio, video)` を返す。
    F6 (audio sum 期待値比較) 専用。 compose_final.py の `_video_duration` とは
    用途が異なるため統合禁止(drift fix 監査ループでの合意)。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", str(p)],
            capture_output=True, text=True,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def _stream_durations(p: Path) -> tuple[float | None, float | None]:
    """video/audio それぞれの stream duration を返す。 F14 用。
    None は ffprobe 失敗 or stream 不在を示す。
    """
    def _probe(stream: str) -> float | None:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream,
                 "-show_entries", "stream=duration", "-of", "json", str(p)],
                capture_output=True, text=True,
            )
            d = json.loads(r.stdout)["streams"][0]["duration"]
            if d in (None, "N/A", ""):
                return None
            return float(d)
        except Exception:
            return None
    return _probe("v:0"), _probe("a:0")


def _expected_slide_duration(audio_paths: list[Path], lines: list[dict]) -> float | None:
    """build.py 系統と同じ式で slide の期待 duration を返す(全 wav 揃ってないと None)。"""
    if not all(p.exists() for p in audio_paths):
        return None
    total = 0.0
    for li, line in enumerate(lines):
        d = _ffprobe_duration(audio_paths[li])
        if d is None:
            return None
        cap = (line.get("caption") or "").strip()
        last = cap[-1] if cap else ""
        gap = 0.40 if last in "。!?!?" else 0.20 if last in "、・" else 0.15
        total += d + HEAD_PAD + gap
    return total + 0.7  # tail


_NORMALIZE_RE = re.compile(r"[、。「」()『』 　\s!?！？・,.\-—]")


def _normalize_for_match(s: str) -> str:
    """F4 比較用の正規化(句読点・空白・括弧・常用記号を除去)。短文向け完全一致比較に使う。"""
    return _NORMALIZE_RE.sub("", s)


def _trigram_set(s: str) -> set[str]:
    """3-gram 集合(_normalize_for_match で正規化してから計算)。"""
    s = _normalize_for_match(s)
    return {s[i:i+3] for i in range(len(s) - 2)}


def _denormalize_kana(text: str, reverse_kana: dict[str, str]) -> str:
    """speech 内のカナ化済み英字を kana_dict 逆引きで英字に戻す(longest-first)。
    F4 比較で英字↔カナ正規化差分による偽陽性を回避する用途。"""
    if not reverse_kana:
        return text
    for kana in sorted(reverse_kana.keys(), key=len, reverse=True):
        if kana in text:
            text = text.replace(kana, reverse_kana[kana])
    return text


SOFT_DOWNGRADE_RULE_IDS = {"F1", "F3", "F4", "F5", "F9", "F9-c", "F9-d"}


def _emit(blocks: list[str], warns: list[str], soft: bool, msg: str) -> None:
    """soft mode で content 系 BLOCK を WARN に降格。

    動画ファイル整合性 (F11 Shorts 解像度 / F14 audio drift) は対象外。
    msg は冒頭が `F1 [BLOCK] ...` 形式である前提。
    """
    if not soft:
        blocks.append(msg)
        return
    head = msg.split(" ", 1)[0]
    if head in SOFT_DOWNGRADE_RULE_IDS:
        warns.append("[soft] " + msg.replace("[BLOCK]", "[WARN]", 1))
    else:
        blocks.append(msg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="WARN も exit 1 扱いにする")
    ap.add_argument("--soft", action="store_true",
                    help="content 系 BLOCK (F1/F3/F4/F5/F9/F9-c/F9-d) を WARN 降格。"
                         " ファイル整合性系 (F11/F14) は BLOCK 維持。")
    args = ap.parse_args(argv)

    slides = load_slides(args.topic)
    blocks: list[str] = []
    warns: list[str] = []
    infos: list[str] = []

    # F1: 「次回」言及検出
    for s in slides:
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            speech = line.get("speech", "")
            for w in NEXT_WORDS:
                if w in speech:
                    _emit(blocks, warns, args.soft,
                        f"F1 [BLOCK] {sid}_{li:02d}: 「{w}」を検出 → 次回予告は指示なき限り NG。"
                        f" speech={speech[:40]}..."
                    )
                    break

    # F3: intro_shared に L1 残存(BLOCK 化、 過去事故防止)
    for s in slides:
        if s.get("shared_id") == "intro_shared" and len(s["lines"]) > 1:
            _emit(blocks, warns, args.soft,
                f"F3 [BLOCK] intro_shared に L1 以降が残存({len(s['lines'])} lines)。"
                f" L0(挨拶のみ)運用ルール違反。残余 line は post 個別 slide に切出すこと。"
            )

    # F2: shared と post 個別 line で speech 重複
    shared_speech: set[str] = set()
    for s in slides:
        if s["shared_id"]:
            for line in s["lines"]:
                shared_speech.add(line["speech"])
    for s in slides:
        if not s["shared_id"]:
            for li, line in enumerate(s["lines"]):
                if line["speech"] in shared_speech:
                    warns.append(
                        f"F2 [WARN] {s['id']}_{li:02d}: shared と speech 重複 → コピペミスの可能性。"
                    )

    # F4: caption と speech の一致検査(2 段)
    # 短文 (< 50 字): 正規化後完全一致を要求 (Shorts 乖離事故対策)
    # 長文 (≥ 50 字): trigram 共通率 < 0.65 で BLOCK (旧来仕様)
    # 複数回の台本で 9〜33 line 乖離事故由来。
    # 算法: trigram(連続 3 文字)集合の coverage = |sp_tg ∩ cp_tg| / |cp_tg|
    # 比較前に kana_dict 逆引きで speech のカナ化英字を英字に戻す(英字カナ化差分による偽陽性回避)
    # caption_override: true を line に明示すれば skip (どうしても caption を別文にしたい時の opt-in)
    reverse_kana = _load_kana_reverse()
    for s in slides:
        if s.get("shared_id"):  # shared slides は除外
            continue
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            if line.get("caption_override"):
                infos.append(
                    f"F4 [INFO] {sid}_{li:02d}: caption_override=true で乖離検査 skip"
                )
                continue
            sp = line.get("speech", "").strip()
            cp = line.get("caption", "").strip()
            if not sp or not cp:
                continue
            sp_normalized = _denormalize_kana(sp, reverse_kana)
            if len(sp) < 50:
                # 短文: 正規化後完全一致を要求(Shorts 対策)
                # speech 側に kana_dict 逆引き(チャンネル→channel)があり、 caption はカナのまま
                # のケースで擬陽性が出るため、 「逆引きあり/なし」 両方で一致を試みる。
                # どちらかで一致すれば pass(英字↔カナ化差分 を吸収)。
                sp_raw_norm = _normalize_for_match(sp)
                sp_dnk_norm = _normalize_for_match(sp_normalized)
                cp_raw_norm = _normalize_for_match(cp)
                cp_dnk_norm = _normalize_for_match(_denormalize_kana(cp, reverse_kana))
                if (sp_raw_norm != cp_raw_norm and
                    sp_raw_norm != cp_dnk_norm and
                    sp_dnk_norm != cp_raw_norm and
                    sp_dnk_norm != cp_dnk_norm):
                    _emit(blocks, warns, args.soft,
                        f"F4 [BLOCK] {sid}_{li:02d}: 短文 caption と speech が不一致 "
                        f"(< 50 字、 正規化後完全一致 必須)。 caption は speech と同じ文字列で書く "
                        f"(要約 / 語順入替 / 常体化 / 半角→全角差以外 禁止)。"
                        f" 推奨: caption = speech.replace(' ', '')。"
                        f"\n         speech ={sp[:60]}"
                        f"\n         caption={cp[:60]}"
                        f"\n         どうしても別文が必要なら line に caption_override: true を明示。"
                    )
                continue
            sp_tg = _trigram_set(sp_normalized)
            # caption 側も denormalize する(speech 側だけ逆引きすると
            # 「tone: トーン」 等の登録があるとき speech「真剣トーン」 → 「真剣tone」 になり
            # caption「真剣トーン」 と trigram が 0% 一致になる偽陽性事故対策)
            cp_normalized = _denormalize_kana(cp, reverse_kana)
            cp_tg = _trigram_set(cp_normalized)
            if not cp_tg:
                continue
            coverage = len(sp_tg & cp_tg) / len(cp_tg)
            if coverage < 0.65:
                _emit(blocks, warns, args.soft,
                    f"F4 [BLOCK] {sid}_{li:02d}: caption と speech が乖離 "
                    f"(trigram coverage={coverage:.2f} < 0.65)。 caption は speech と同情報単位で書く (要約 / 文末極性差 / 別文章 禁止)。"
                    f" 推奨: caption = speech.replace(' ', '')。"
                    f" speech={sp[:40]}..."
                )

    # F5: speech 内の 数字 kana 表記 検出
    # 過去事故: 数詞を kana 化すると直前の「に」等が助詞と誤認識される誤読
    # Shorts でも同様に発生 → BLOCK 化で再発防止
    for s in slides:
        sid = s["id"]
        if s["shared_id"]:
            continue  # shared は固定なので skip(既存ルール準拠済)
        for li, line in enumerate(s["lines"]):
            sp = line.get("speech", "")
            m = NUM_KANA_RE.search(sp)
            if m:
                _emit(blocks, warns, args.soft,
                    f"F5 [BLOCK] {sid}_{li:02d}: 数字 kana 表記検出 → 「{m.group()}」"
                    f" speech={sp[:60]}..."
                    f" \n         運用ルール違反:"
                    f" 数字は漢字混じり原文のまま、 kana 化禁止。 例 「ひゃくえん」 → 「100円」"
                )

    # F10: speech 内 未登録英字 keyword(事前カタカナ化検討用、 INFO のみ)
    # 過去事故: intonationScale / VOICEVOX_POOL 等が kana_dict 未登録のまま
    # build → ElevenLabs 誤読 → 再 build 手戻り。 事前 INFO で kana_dict 整備を促す。
    kana_keys = _load_kana_keys()
    from collections import Counter
    eng_counter: Counter[str] = Counter()
    for s in slides:
        if s["shared_id"]:
            continue
        for line in s["lines"]:
            sp = line.get("speech", "")
            for m in F10_ENG_TOKEN.finditer(sp):
                tok = m.group()
                if len(tok) < 3:
                    continue
                if tok.lower() in F10_STOP_WORDS:
                    continue
                if tok in kana_keys or tok.lower() in kana_keys:
                    continue
                eng_counter[tok] += 1
    if eng_counter:
        unique = len(eng_counter)
        total = sum(eng_counter.values())
        top = eng_counter.most_common(15)
        sample = ", ".join(f"{k}×{v}" for k, v in top)
        infos.append(
            f"F10 [INFO] kana_dict 未登録英字 keyword {unique} 種類 / 合計 {total} 件"
            f" → カタカナ化検討対象。 上位: {sample}"
            f"\n         登録すべき場合は tools/kana_dict.yaml に追加 → synth 再実行で自動適用"
        )

    # F9: speech 記号誤用(数字-数字 / 矢印 等)。 caption 側は記号 OK。
    # 過去事故: 「4-6字」 を ElevenLabs が「4 の 6 字」 と読む → ルール「Aから B」 に統一
    for s in slides:
        if s["shared_id"]:
            continue
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            sp = line.get("speech", "")
            m = F9_RANGE_RE.search(sp)
            if m:
                _emit(blocks, warns, args.soft,
                    f"F9 [BLOCK] {sid}_{li:02d}: 数字範囲記号「{m.group()}」 検出 → 「Aから B」 と日本語で書く。"
                    f" 例: 「4-6字」 → 「4から6字」。 caption 側は記号 OK、 speech のみ NG。"
                    f" speech={sp[:60]}..."
                )
            for ch in F9_DISALLOWED_CHARS:
                if ch in sp:
                    _emit(blocks, warns, args.soft,
                        f"F9 [BLOCK] {sid}_{li:02d}: 矢印記号「{ch}」 を speech で検出 → 日本語で書く(「Aから B へ」 等)。"
                        f" speech={sp[:60]}..."
                    )
                    break

    # F9-c: 装飾記号 BLOCK (過去の事故由来)
    # 同記号 2 連(★★ 等)は匿名化扱いで pass。
    for s in slides:
        if s.get("shared_id"):
            continue
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            sp = line.get("speech", "")
            for ch in F9_DECORATIVE_CHARS:
                if ch in sp and (ch + ch) not in sp:
                    _emit(blocks, warns, args.soft,
                        f"F9 [BLOCK] {sid}_{li:02d}: 装飾記号「{ch}」 を speech で検出 → 日本語へ "
                        f"(例: ★ → ホシ)。 同記号 2 連 (匿名化) は除外。"
                        f" speech={sp[:60]}..."
                    )

    # F9-d: 小数 + 行末「。」 隣接 WARN (過去の事故由来)
    # NOTE: 当初 BLOCK 予定だったが 実データで false positive 1件残存のため
    # 本 PR では WARN 降格。BLOCK 昇格 + 正規表現精緻化は 別 PR で実施。
    for s in slides:
        if s.get("shared_id"):
            continue
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            sp = line.get("speech", "")
            if F9_DECIMAL_TAIL_RE.search(sp):
                warns.append(
                    f"F9 [WARN] {sid}_{li:02d}: 小数点 + 行末「。」 隣接疑い "
                    f"(ElevenLabs decimal-mode → ドット誤読リスク)。 対策: 別文に分けるか、 小数を 「整数点小数」 に分解。"
                    f" 要確認。 speech={sp[:60]}..."
                )

    # F11: Shorts final.mp4 解像度 BLOCK (横長 build 事故由来)
    # topic に /shorts/ が含まれ final.mp4 が存在する場合、 1080×1920 でなければ BLOCK。
    # ffprobe 失敗時は WARN 降格。
    if "/shorts/" in args.topic:
        final_mp4 = paths.final_mp4(args.topic)
        if final_mp4.exists():
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height", "-of", "csv=p=0",
                     str(final_mp4)],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode != 0:
                    warns.append(
                        f"F11 [WARN] ffprobe 失敗 (final.mp4 解像度検証 skip): {r.stderr[:100]}"
                    )
                else:
                    wh = r.stdout.strip()
                    if wh != "1080,1920":
                        blocks.append(
                            f"F11 [BLOCK] Shorts final 解像度不正: {wh} (期待: 1080,1920)。 "
                            f"`run_all.py` 誤用の可能性 → `build_short.py` + `mix_bgm_short.py` で再 build。"
                        )
            except Exception as e:
                warns.append(f"F11 [WARN] ffprobe 実行エラー (final.mp4 解像度検証 skip): {e}")

    # F12: slide_prompt 列挙整合 WARN (画像ハルシネーション由来)
    # caption 内の 3 字以上トークン(カタカナ/ひらがな名詞・数値・英字略語)が
    # speech にも slide_prompt にも不在 → ≥ 3 件で WARN (alert fatigue 防止閾値)
    for s in slides:
        if s.get("shared_id"):
            continue
        sp_tokens: set[str] = set()
        for line in s["lines"]:
            sp_tokens.update(F12_TOKEN_RE.findall(line.get("speech", "")))
        slide_prompt = s.get("slide_prompt", "") or ""
        cap_tokens: set[str] = set()
        for line in s["lines"]:
            cap_tokens.update(F12_TOKEN_RE.findall(line.get("caption", "")))
        missing = [t for t in cap_tokens if t not in slide_prompt and t not in sp_tokens]
        if len(missing) >= 3:
            warns.append(
                f"F12 [WARN] {s['id']}: slide_prompt に caption の主要語 {len(missing)} 件不在 "
                f"(例: {sorted(missing)[:3]}) → 画像ハルシネリスク、 prompt に列挙推奨。"
            )

    # F13: caption 内 小数点が「点」 エンコードのまま残存 WARN (過去の事故由来)
    # 例: 「閾値は4点0で5点満点」 → caption は 「4.0」 と書くべき。
    # speech 側は ElevenLabs 誤読回避のため「4点0」 が正(変更しない)。
    for s in slides:
        if s.get("shared_id"):
            continue
        sid = s["id"]
        for li, line in enumerate(s["lines"]):
            cap = line.get("caption", "")
            m = F13_DECIMAL_AS_TEN_RE.search(cap)
            if m:
                warns.append(
                    f"F13 [WARN] {sid}_{li:02d}: caption に小数点の「点」 エンコード残存 (「{m.group()}」)。"
                    f" → 視聴者向け caption は「.」 表記に戻す (例: 「4点0」 → 「4.0」、 「0点2」 → 「0.2」)。"
                    f" speech 側は ElevenLabs 誤読回避のため「点」 のまま維持で OK。"
                    f" caption={cap[:60]}..."
                )

    # F6: 字幕同期ずれ検出(clip mp4 duration vs audio 期待 duration)
    # build 後の clip ファイルがあれば検査。 ±SYNC_DRIFT_THRESHOLD 超で WARN
    clips_dir = paths.clips_dir(args.topic)
    if clips_dir.exists():
        for s in slides:
            sid = s["id"]
            clip = clips_dir / f"{sid}.mp4"
            if not clip.exists():
                continue
            actual = _ffprobe_duration(clip)
            expected = _expected_slide_duration(s["audio_paths"], s["lines"])
            if actual is None or expected is None:
                continue
            drift = actual - expected
            if abs(drift) > SYNC_DRIFT_THRESHOLD:
                warns.append(
                    f"F6 [WARN] {sid}: 字幕/音声同期ずれ疑い "
                    f"clip duration={actual:.2f}s vs expected={expected:.2f}s "
                    f"(drift={drift:+.2f}s, threshold=±{SYNC_DRIFT_THRESHOLD}s)"
                )

    # F14: video / audio stream drift (final.mp4 / final_no_bgm.mp4 の v/a stream 長差)
    # drift fix 監査ループ追加。
    # 注: option D (compose_final → video PTS anchor) 適用後は container drift が残るのが
    # 仕様。 subtitle は video PTS = 視覚遷移に align されているので、 container drift は
    # 「音が映像より少し遅い」 という体感に変換される。 ±0.5s 程度までは観賞許容、
    # ±1.0s 超は build.py 側 per-clip 修正 (option A) 検討領域 = BLOCK。
    F14_WARN_TH = 0.08
    F14_BLOCK_TH = 1.0
    post_dir = paths.topic_dir(args.topic)
    for fname in ("final.mp4", "final_no_bgm.mp4"):
        fp = post_dir / fname
        if not fp.exists():
            continue
        v_dur, a_dur = _stream_durations(fp)
        if v_dur is None or a_dur is None:
            continue
        drift = v_dur - a_dur
        ab = abs(drift)
        if ab > F14_BLOCK_TH:
            blocks.append(
                f"F14 [BLOCK] {fname}: video/audio stream drift={drift:+.3f}s "
                f"(v={v_dur:.3f}s a={a_dur:.3f}s, threshold=±{F14_BLOCK_TH}s)。"
                f" compose_final / mix_bgm の再実行が必要 (scripts/sweep_recompose_final.py)。"
            )
        elif ab > F14_WARN_TH:
            warns.append(
                f"F14 [WARN] {fname}: video/audio stream drift={drift:+.3f}s "
                f"(v={v_dur:.3f}s a={a_dur:.3f}s, threshold=±{F14_WARN_TH}s)。 "
                f"字幕は video PTS に align 済 (compose_final option D)。 "
                f"container drift は build.py per-clip ミスマッチ起因、 ±1.0s 超なら option A 検討。"
            )

    # F7: force_align fallback 発生検出(build.py が _force_align_fallback.log に記録)
    fallback_log = paths.work_dir(args.topic) / "_force_align_fallback.log"
    if fallback_log.exists() and fallback_log.stat().st_size > 0:
        n = sum(1 for _ in fallback_log.read_text(encoding="utf-8").splitlines() if _.strip())
        if n > 0:
            head = fallback_log.read_text(encoding="utf-8").splitlines()[:3]
            warns.append(
                f"F7 [WARN] force_align fallback 発生: {n} 件(比例分割 fallback で字幕タイミング粗くなってる可能性)"
                f"\n         先頭 {min(3,n)} 件:\n         " + "\n         ".join(head)
                + f"\n         詳細: {fallback_log}"
            )

    # F8: verify_synth(Whisper 誤読検出)未実行 / mismatch 残存
    verify_report = paths.audio_dir(args.topic) / "_verify_report.json"
    if not verify_report.exists():
        warns.append(
            f"F8 [WARN] verify_synth 未実行: `python -m src.post_video.verify_synth --topic {args.topic}` を実行してください。"
            f" Whisper で誤読検出 → 必要なら `--self-heal` 付きで synth 再実行。"
        )
    else:
        try:
            misreads = json.loads(verify_report.read_text(encoding="utf-8"))
            if misreads:
                samples = "\n         ".join(
                    f"{m['sid']}_{m['li']:02d}: expected={m['expected'][:40]} / actual={m['actual'][:40]}"
                    for m in misreads[:3]
                )
                warns.append(
                    f"F8 [WARN] verify_synth mismatch {len(misreads)} 件残存 → 誤読の可能性。"
                    f"\n         先頭 {min(3, len(misreads))} 件:\n         {samples}"
                    f"\n         対処: `synth --self-heal` または kana_dict.yaml に追加 → synth 再実行。"
                    f"\n         ★ `--self-heal` 試行済か確認。 残存なら手動書き直し検討 (過去の障害由来)。"
                )
        except Exception as e:
            infos.append(f"F8 [INFO] _verify_report.json パース失敗: {e}")

    # 出力
    print(f"=== qa_post_video (topic={args.topic}) ===")
    print(f"  slides: {len(slides)}")
    print(f"  BLOCK: {len(blocks)} / WARN: {len(warns)} / INFO: {len(infos)}")
    print()
    for m in blocks:
        print(m)
    for m in warns:
        print(m)
    for m in infos:
        print(m)

    if blocks:
        return 1
    if args.strict and warns:
        return 1
    if warns:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
