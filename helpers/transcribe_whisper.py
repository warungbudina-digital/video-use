#!/usr/bin/env python3
"""Offline transcription shim: whisper.cpp -> Scribe-shaped transcript JSON.

ADDED FILE (not upstream browser-use/video-use). Lets you skip ElevenLabs
Scribe entirely: no API key, no network, audio never leaves the device.
It reproduces only the *output* of helpers/transcribe.py that
helpers/pack_transcripts.py actually consumes.

Output: <edit_dir>/transcripts/<video_stem>.json  (same path as transcribe.py)
Shape (verified against pack_transcripts.group_into_phrases):
    {"words": [
        {"type":"word","text":str,"start":<sec>,"end":<sec>,"speaker_id":"speaker_0"},
        ...
    ]}
Only `type`,`text`,`start`,`end`,`speaker_id` are read downstream.
whisper.cpp has no diarization  -> every word is speaker_0.
whisper.cpp has no audio events -> none emitted (pack_transcripts tolerates that).

Usage:
    python helpers/transcribe_whisper.py <video_or_dir> [--edit-dir DIR]
        [--model ~/models/ggml-base.en.bin] [--whisper /path/to/whisper-cli]
        [--language en] [--threads 4] [--audio-track 0] [--force]

Prereqs (Termux):
    pkg install ffmpeg whisper.cpp
    # a ggml model, e.g.:
    mkdir -p ~/models && curl -L -o ~/models/ggml-base.en.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
    # better word timing (optional): pass --dtw via WHISPER_EXTRA below and use a DTW-capable model.

Then feed the editor as usual:
    python helpers/pack_transcripts.py --edit-dir <edit_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv"}
# whisper.cpp special tokens: [_BEG_], [_TT_123], [_SOT_], [_EOT_], [BLANK_AUDIO], (music) markers...
SPECIAL_TOK = re.compile(r"^(\[_.*_\]|\[[A-Z_]+\])$")
# extra flags for whisper (e.g. "--dtw base.en") via env, kept out of the arg surface
WHISPER_EXTRA = os.environ.get("WHISPER_EXTRA", "").split()


def find_whisper(explicit: str | None) -> str:
    if explicit:
        if not shutil.which(explicit) and not Path(explicit).exists():
            sys.exit(f"whisper binary not found: {explicit}")
        return explicit
    for name in ("whisper-cli", "whisper", "main"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit("whisper.cpp binary not found (looked for whisper-cli/whisper/main). "
             "Install with `pkg install whisper.cpp`, or pass --whisper /path.")


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Mirror of helpers/transcribe.py:transcript_path so the two never drift."""
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def extract_audio(video: Path, wav: Path, track: int) -> None:
    # 16 kHz mono PCM s16le = whisper.cpp's required input format.
    cmd = ["ffmpeg", "-y", "-i", str(video),
           "-map", f"0:a:{track}?", "-vn",
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_whisper(whisper: str, wav: Path, model: Path,
                language: str | None, threads: int, out_prefix: Path) -> Path:
    cmd = [whisper, "-m", str(model), "-f", str(wav),
           "-t", str(threads), "-np",
           "--output-json-full", "--output-file", str(out_prefix)]
    if language:
        cmd += ["-l", language]
    cmd += WHISPER_EXTRA
    subprocess.run(cmd, check=True)
    j = Path(str(out_prefix) + ".json")
    if j.exists():
        return j
    alt = Path(str(wav) + ".json")  # some builds name it after the input
    if alt.exists():
        return alt
    sys.exit(f"whisper JSON not produced (expected {j})")


def _ms_to_s(ms) -> float:
    return round(float(ms) / 1000.0, 3)


def _tokens_to_words(tokens: list[dict]) -> list[dict]:
    """Merge whisper subword tokens into words on the leading-space convention.

    A token whose text starts with a space begins a new word; tokens without a
    leading space (subword pieces, punctuation) attach to the current word.
    Word start = first token's offset.from, end = last token's offset.to.
    """
    words: list[dict] = []
    cur: dict | None = None
    for tk in tokens:
        raw = tk.get("text", "")
        if SPECIAL_TOK.match(raw.strip()):
            continue
        off = tk.get("offsets") or {}
        a, b = off.get("from"), off.get("to")
        if a is None or b is None or a < 0 or b < 0:
            continue
        piece = raw.strip()
        if piece == "":
            continue
        starts_word = raw.startswith(" ") or cur is None
        if starts_word:
            if cur:
                words.append(cur)
            cur = {"text": piece, "start": a, "end": b}
        else:
            cur["text"] += piece
            cur["end"] = b
    if cur:
        words.append(cur)
    return words


def parse_whisper(jpath: Path) -> list[dict]:
    data = json.loads(jpath.read_text())
    segments = data.get("transcription", []) or []
    words: list[dict] = []
    for seg in segments:
        toks = seg.get("tokens")
        if toks:
            for w in _tokens_to_words(toks):
                words.append({"type": "word", "text": w["text"],
                              "start": _ms_to_s(w["start"]), "end": _ms_to_s(w["end"]),
                              "speaker_id": "speaker_0"})
        else:
            # fallback: no token timings in this build -> one entry per segment
            off = seg.get("offsets") or {}
            a, b = off.get("from"), off.get("to")
            txt = (seg.get("text") or "").strip()
            if txt and a is not None and b is not None:
                words.append({"type": "word", "text": txt,
                              "start": _ms_to_s(a), "end": _ms_to_s(b),
                              "speaker_id": "speaker_0"})
    return words


def transcribe_one(video: Path, edit_dir: Path, whisper: str, model: Path,
                   language: str | None, threads: int, audio_track: int,
                   force: bool, verbose: bool = True) -> Path:
    out_path = transcript_path(edit_dir, video, audio_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{video.stem}.wav"
        if verbose:
            print(f"  extracting audio: {video.name}", flush=True)
        extract_audio(video, wav, audio_track)
        if verbose:
            print(f"  transcribing (whisper.cpp): {wav.name}", flush=True)
        jpath = run_whisper(whisper, wav, model, language, threads, Path(tmp) / video.stem)
        words = parse_whisper(jpath)
    out_path.write_text(json.dumps({"words": words}, indent=2))
    if verbose:
        print(f"  saved: {out_path.name}  ({len(words)} words)")
    return out_path


def iter_videos(target: Path):
    if target.is_dir():
        for p in sorted(target.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        yield target


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline whisper.cpp transcription -> Scribe-shaped JSON")
    ap.add_argument("target", help="video file or directory of videos")
    ap.add_argument("--edit-dir", help="edit dir (default: <target-or-parent>/edit)")
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL", str(Path.home() / "models/ggml-base.en.bin")),
                    help="path to ggml model (default: ~/models/ggml-base.en.bin or $WHISPER_MODEL)")
    ap.add_argument("--whisper", help="path/name of whisper.cpp binary (default: auto-detect)")
    ap.add_argument("--language", default=None, help="language code, e.g. en / id (default: auto)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4)), help="whisper threads")
    ap.add_argument("--audio-track", type=int, default=0, help="0-based audio track")
    ap.add_argument("--force", action="store_true", help="re-transcribe even if cached")
    args = ap.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        sys.exit(f"not found: {target}")
    model = Path(args.model).expanduser()
    if not model.exists():
        sys.exit(f"model not found: {model}\n"
                 f"  download e.g.: curl -L -o {model} "
                 f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin")
    whisper = find_whisper(args.whisper)
    base = target if target.is_dir() else target.parent
    edit_dir = Path(args.edit_dir).expanduser().resolve() if args.edit_dir else base / "edit"

    vids = list(iter_videos(target))
    if not vids:
        sys.exit(f"no videos found in {target}")
    for v in vids:
        transcribe_one(v, edit_dir, whisper, model, args.language,
                       args.threads, args.audio_track, args.force)
    print(f"done -> {edit_dir / 'transcripts'}  "
          f"(next: python helpers/pack_transcripts.py --edit-dir {edit_dir})")


if __name__ == "__main__":
    main()
