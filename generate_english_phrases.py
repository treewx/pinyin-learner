#!/usr/bin/env python3
"""
Generate English mnemonic images from english_words_todo.csv.

Reads the 'your_word' column — which can be a single word OR a multi-word
phrase (e.g. "play ball", "turn on", "thank you") — and generates object-
typography images where the English letters are formed from the mnemonic
objects representing the Chinese pronunciation, coloured by tone.

Usage
-----
1.  Fill in the 'your_word' column in english_words_todo.csv
    (leave blank for any entries you want to skip)
2.  Run:
        $env:OPENAI_API_KEY = "sk-..."
        python generate_english_phrases.py

Resumable: already-generated images are skipped on re-run.
"""

import base64
import csv
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent
MNEMONIC_CSV  = SCRIPT_DIR / "Chinese Mnemonic Source - Sheet1.csv"
TODO_CSV      = SCRIPT_DIR / "english_words_todo.csv"
OUTPUT_DIR    = Path("english_images")        # same folder as single-word images
PROGRESS_FILE = Path("english_progress.json") # shared progress file

# ── Image settings ─────────────────────────────────────────────────────────────

MODEL          = "gpt-image-2"
SIZE           = "1024x1024"
QUALITY        = "low"       # $0.02 / image
JPEG_QUALITY   = 88
DELAY_SECONDS  = int(os.environ.get("DELAY_SECONDS", "13"))
COST_PER_IMAGE = 0.02

TONE_COLOUR = {
    1: "golden yellow",
    2: "warm orange",
    3: "bright green",
    4: "red",
    0: "grey",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def to_slug(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = re.sub(r"\d", "", ascii_str)
    return re.sub(r"[^\w]", "_", ascii_str).strip("_").lower()


def load_mnemonics() -> dict[str, str]:
    result: dict[str, str] = {}
    with open(MNEMONIC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        pcol = col.get("pinyin") or col.get("Pinyin", "")
        mcol = col.get("mnemonic") or col.get("visual mnemonic", "")
        for row in reader:
            p = (row.get(pcol) or "").strip()
            m = (row.get(mcol) or "").strip()
            if p and m:
                m = re.sub(r"\s*\(.*?\)", "", m).strip().rstrip(".,")
                result[to_slug(p)] = m
    return result


def get_syllables_with_tones(chinese: str) -> list[tuple[str, int]]:
    try:
        from pypinyin import pinyin as to_py, Style
        result = []
        for py in to_py(chinese, style=Style.TONE3, heteronym=False):
            if py and py[0]:
                syl_t3 = py[0].lower()
                tone   = int(syl_t3[-1]) if syl_t3[-1].isdigit() else 0
                slug   = to_slug(re.sub(r"\d", "", syl_t3))
                result.append((slug, tone))
        return result
    except ImportError:
        return []


def build_prompt(phrase: str,
                 syllables_tones: list[tuple[str, int]],
                 mnemonics: dict[str, str]) -> str | None:
    """
    Build an object-typography prompt for a word or multi-word phrase.
    Spaces are kept in the displayed phrase but skipped when listing letters.
    """
    phrase_upper = phrase.upper()

    # List just the letters (no spaces) with dashes between
    letters = " – ".join(c for c in phrase_upper if c != " ")

    # Deduplicated (mnemonic, colour) pairs
    components: list[tuple[str, str]] = []
    for slug, tone in syllables_tones:
        m = mnemonics.get(slug, "")
        if not m:
            continue
        colour = TONE_COLOUR.get(tone, "grey")
        pair   = (m, colour)
        if pair not in components:
            components.append(pair)

    if not components:
        return None

    if len(components) == 1:
        m, c = components[0]
        objects_desc = f"{m} coloured {c}"
    else:
        parts = [f"{m} coloured {c}" for m, c in components]
        objects_desc = ", ".join(parts[:-1]) + ", and " + parts[-1]

    return (
        f'Object typography artwork: the English word "{phrase_upper}" spelled in '
        f"large bold block letters, where every single letter ({letters}) is physically "
        f"constructed by arranging and bending the following objects into letter shapes: "
        f"{objects_desc}. "
        f"Each letter must be clearly legible and formed entirely from these objects. "
        f"Flat white background. Centered composition. "
        f"Clean graphic illustration style with high contrast. "
        f'No extra text, no other objects — just "{phrase_upper}" built from {objects_desc}.'
    )


def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")


def save_jpeg(b64_data: str, path: Path) -> None:
    img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
    img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    for p in (MNEMONIC_CSV, TODO_CSV):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    try:
        from pypinyin import pinyin  # noqa: F401
    except ImportError:
        sys.exit("pypinyin is required.  Run:  pip install pypinyin")

    client    = OpenAI()
    mnemonics = load_mnemonics()
    done      = load_progress()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Read todo CSV ─────────────────────────────────────────────────────────
    entries: list[dict] = []
    skipped_blank = 0

    with open(TODO_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            chinese          = (row.get("chinese")          or "").strip()
            your_word        = (row.get("your_word")        or "").strip()
            original_english = (row.get("original_english") or "").strip()
            pinyin_csv       = (row.get("pinyin")           or "").strip()

            # If no override supplied, fall back to the default English
            if not your_word:
                your_word = original_english.split("/")[0].strip()

            slug = to_slug(your_word)
            if not slug or slug in done:
                continue

            # Get syllables from Chinese characters via pypinyin
            chars = "".join(c for c in chinese if "一" <= c <= "鿿")
            if not chars:
                continue
            syllables_tones = get_syllables_with_tones(chars)
            if not syllables_tones:
                continue

            prompt = build_prompt(your_word, syllables_tones, mnemonics)
            if not prompt:
                continue

            entries.append({
                "phrase":          your_word,
                "chinese":         chinese,
                "slug":            slug,
                "syllables_tones": syllables_tones,
                "prompt":          prompt,
            })

    # ── Header ────────────────────────────────────────────────────────────────
    bar  = "─" * 64
    cost = len(entries) * COST_PER_IMAGE
    mins = len(entries) * DELAY_SECONDS / 60
    print(bar)
    print("  English phrase mnemonic generator  (from english_words_todo.csv)")
    print(bar)
    print(f"  Rows with your_word filled : {len(entries) + len(done)}")
    print(f"  Rows left blank (skip)     : {skipped_blank}")
    print(f"  Already generated          : {len(done)}")
    print(f"  To generate                : {len(entries)}")
    print(f"  Estimated cost             : ${cost:.2f}  (@ $0.02/image)")
    print(f"  Estimated time             : ~{mins:.0f} min  (delay={DELAY_SECONDS}s)")
    print(f"  Output folder              : {OUTPUT_DIR.resolve()}")
    print(bar)

    if not entries:
        print("  Nothing to do!")
        return

    # Show a preview
    print("  Preview (first 5):")
    for e in entries[:5]:
        syls = " + ".join(
            f"{sl}({TONE_COLOUR.get(t,'grey')[:3]})" for sl, t in e["syllables_tones"]
        )
        print(f"    {e['phrase']:<20} {e['chinese']}  →  {syls}")
    print()

    answer = input("  Continue? [y/N] ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        sys.exit(0)
    print()

    errors: list[dict] = []

    for i, entry in enumerate(entries, 1):
        phrase   = entry["phrase"]
        chinese  = entry["chinese"]
        slug     = entry["slug"]
        prompt   = entry["prompt"]
        out_path = OUTPUT_DIR / f"{slug}.jpg"

        syls = " + ".join(
            f"{sl}({TONE_COLOUR.get(t,'grey')[:3]})" for sl, t in entry["syllables_tones"]
        )
        print(f"[{i:>3}/{len(entries)}]  {phrase:<22} {chinese:<6}  {syls}")

        try:
            response = client.images.generate(
                model=MODEL, prompt=prompt, size=SIZE, quality=QUALITY, n=1,
            )
            img_data = response.data[0]
            if img_data.b64_json:
                save_jpeg(img_data.b64_json, out_path)
            else:
                r = requests.get(img_data.url, timeout=60)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

            done.add(slug)
            save_progress(done)
            print(f"           ✓  {out_path.name}")

        except Exception as exc:
            err_msg = str(exc)
            print(f"           ✗  ERROR: {err_msg[:80]}")
            errors.append({"phrase": phrase, "error": err_msg})
            if "rate" in err_msg.lower() or "429" in err_msg:
                print("           ⏸  Rate limited — waiting 60 s extra")
                time.sleep(60)

        if i < len(entries):
            time.sleep(DELAY_SECONDS)

    print()
    print(f"Done. {len(done)} total images in {OUTPUT_DIR}/")

    if errors:
        log = Path("english_phrase_errors.json")
        log.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"⚠  {len(errors)} errors → {log}")
        print("   Re-run to retry.")


if __name__ == "__main__":
    main()
