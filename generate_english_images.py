#!/usr/bin/env python3
"""
Generate English object-typography mnemonic images for HSK 1 vocabulary.

For each single-word HSK 1 entry the English word is rendered as large block
letters where every letter is physically formed from the visual mnemonic
objects that represent the Chinese pinyin syllables — coloured to match
their tone (yellow=1st, orange=2nd, green=3rd, red=4th, grey=neutral).

Example
-------
  "company"  →  公司  →  gōng (tone 1) + sī (tone 1)
  →  COMPANY built from golden-yellow gong instruments and golden-yellow soot

Usage
-----
    Set your API key:
        $env:OPENAI_API_KEY = "sk-..."
    Then run:
        python generate_english_images.py

Resumable: already-generated images are skipped on re-run.
Multi-word English entries (e.g. "thank you") are skipped.
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
HSK_CSVS      = [SCRIPT_DIR / "hsk1.csv", SCRIPT_DIR / "hsk2.csv"]
OUTPUT_DIR    = Path("english_images")
PROGRESS_FILE = Path("english_progress.json")

# ── Image settings ─────────────────────────────────────────────────────────────

MODEL          = "gpt-image-2"
SIZE           = "1024x1024"
QUALITY        = "low"       # $0.02/image — same as the pinyin image run
JPEG_QUALITY   = 88
DELAY_SECONDS  = int(os.environ.get("DELAY_SECONDS", "13"))
COST_PER_IMAGE = 0.02

# ── Tone → colour name (used in the image prompt) ──────────────────────────────

TONE_COLOUR = {
    1: "golden yellow",
    2: "warm orange",
    3: "bright green",
    4: "red",
    0: "grey",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def to_slug(s: str) -> str:
    """Strip diacritics/digits and make filename-safe."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = re.sub(r"\d", "", ascii_str)
    return re.sub(r"[^\w]", "_", ascii_str).strip("_").lower()


def load_mnemonics() -> dict[str, str]:
    """Return {pinyin_slug: mnemonic_text} from the mnemonic CSV.
    Handles both column-name variants ('Pinyin'/'pinyin', 'Mnemonic'/'visual mnemonic').
    """
    result: dict[str, str] = {}
    with open(MNEMONIC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Build a lowercase → actual column name map
        col = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        pinyin_col   = col.get("pinyin") or col.get("Pinyin", "")
        mnemonic_col = col.get("mnemonic") or col.get("visual mnemonic", "")
        for row in reader:
            pinyin   = (row.get(pinyin_col)   or "").strip()
            mnemonic = (row.get(mnemonic_col) or "").strip()
            if pinyin and mnemonic:
                mnemonic = re.sub(r"\s*\(.*?\)", "", mnemonic).strip().rstrip(".,")
                result[to_slug(pinyin)] = mnemonic
    return result


def get_syllables_with_tones(chinese: str) -> list[tuple[str, int]]:
    """
    Use pypinyin to get (slug, tone_number) for each Chinese character.
    Returns an empty list if pypinyin is unavailable.
    """
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


def build_prompt(english_word: str,
                 syllables_tones: list[tuple[str, int]],
                 mnemonics: dict[str, str]) -> str | None:
    """
    Build an object-typography prompt for the English word.

    Each unique (mnemonic, tone-colour) pair is listed separately so the
    image model knows to render different syllables in different colours.
    """
    word    = english_word.upper()
    letters = " – ".join(word)

    # Build deduplicated list of (mnemonic_text, colour_name)
    components: list[tuple[str, str]] = []
    for slug, tone in syllables_tones:
        mnemonic = mnemonics.get(slug, "")
        if not mnemonic:
            continue
        colour = TONE_COLOUR.get(tone, "grey")
        pair   = (mnemonic, colour)
        if pair not in components:
            components.append(pair)

    if not components:
        return None

    # Natural-language description of objects + colours
    if len(components) == 1:
        m, c = components[0]
        objects_desc = f"{m} coloured {c}"
    else:
        parts = [f"{m} coloured {c}" for m, c in components]
        objects_desc = ", ".join(parts[:-1]) + ", and " + parts[-1]

    return (
        f'Object typography artwork: the English word "{word}" spelled in large bold '
        f"block letters, where every single letter ({letters}) is physically constructed "
        f"by arranging and bending the following objects into letter shapes: {objects_desc}. "
        f"Each letter must be clearly legible and formed entirely from these objects. "
        f"Flat white background. Centered composition. "
        f"Clean graphic illustration style with high contrast. "
        f'No extra text, no other objects — just the word "{word}" '
        f"built from {objects_desc}."
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
    # Sanity checks
    for p in [MNEMONIC_CSV] + HSK_CSVS:
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

    # ── Build entry list from all HSK CSVs ───────────────────────────────────
    entries: list[dict] = []
    seen_slugs: set[str] = set()

    for csv_path in HSK_CSVS:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader   = csv.DictReader(f)
            fieldmap = {c.strip().lower(): c for c in (reader.fieldnames or [])}

            hanzi_col   = fieldmap.get("hanzi") or fieldmap.get("chinese", "")
            english_col = fieldmap.get("english", "")

            for row in reader:
                chinese = (row.get(hanzi_col) or "").strip()
                english = (row.get(english_col) or "").strip()
                if not chinese or not english:
                    continue

                # Use the first variant before "/"
                english = english.split("/")[0].strip()

                # ── Skip multi-word English entries ───────────────────────────
                if len(english.split()) > 1:
                    continue

                slug = to_slug(english)
                if not slug or slug in seen_slugs:
                    continue
                seen_slugs.add(slug)

                if slug in done:
                    continue

                syllables_tones = get_syllables_with_tones(chinese)
                if not syllables_tones:
                    continue

                prompt = build_prompt(english, syllables_tones, mnemonics)
                if not prompt:
                    continue

                entries.append({
                    "english":         english,
                    "chinese":         chinese,
                    "slug":            slug,
                    "syllables_tones": syllables_tones,
                    "prompt":          prompt,
                })

    # ── Print header ──────────────────────────────────────────────────────────
    bar  = "─" * 64
    cost = len(entries) * COST_PER_IMAGE
    mins = len(entries) * DELAY_SECONDS / 60
    print(bar)
    print("  English mnemonic image generator  (HSK 1 + 2, single-word only)")
    print(bar)
    print(f"  Eligible entries : {len(seen_slugs)}")
    print(f"  Already done     : {len(done)}")
    print(f"  To generate      : {len(entries)}")
    print(f"  Estimated cost   : ${cost:.2f}  "
          f"(gpt-image-2 low @ $0.02/image)")
    print(f"  Estimated time   : ~{mins:.0f} min  (delay={DELAY_SECONDS}s)")
    print(f"  Output folder    : {OUTPUT_DIR.resolve()}")
    print(bar)

    if not entries:
        print("  Nothing to do — all images already generated!")
        return

    answer = input("  Continue? [y/N] ").strip().lower()
    if answer != "y":
        print("  Aborted.")
        sys.exit(0)
    print()

    errors: list[dict] = []

    for i, entry in enumerate(entries, 1):
        english  = entry["english"]
        chinese  = entry["chinese"]
        slug     = entry["slug"]
        prompt   = entry["prompt"]
        out_path = OUTPUT_DIR / f"{slug}.jpg"

        # Summary line: english  Chinese  syl(colour) + syl(colour) ...
        syl_summary = " + ".join(
            f"{sl}({TONE_COLOUR.get(t, 'grey')[:3]})"
            for sl, t in entry["syllables_tones"]
        )
        print(f"[{i:>3}/{len(entries)}]  {english:<20} {chinese:<6}  {syl_summary}")

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
            errors.append({"english": english, "error": err_msg})
            if "rate" in err_msg.lower() or "429" in err_msg:
                print("           ⏸  Rate limited — waiting 60 s extra")
                time.sleep(60)

        if i < len(entries):
            time.sleep(DELAY_SECONDS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"Done. {len(done)} images saved to {OUTPUT_DIR}/")

    if errors:
        log = Path("english_errors.json")
        log.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"⚠  {len(errors)} errors logged to {log}")
        print("   Re-run the script to retry failed items.")


if __name__ == "__main__":
    main()
