#!/usr/bin/env python3
"""
Pinyin Visual Learner
=====================
Tab 1 – Flashcard study mode for HSK 1 & 2 vocabulary.
         Shows the Chinese word, then your object-typography syllable images
         as memory hooks before revealing the meaning.

Tab 2 – Visual Paragraph Generator.
         Paste any Chinese text and get it rendered as a paragraph of your
         mnemonic images — one image per character/syllable.

Run with:
    streamlit run app.py
"""

import base64
import io
import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

try:
    from pypinyin import pinyin as to_pinyin, Style
    PYPINYIN_OK = True
except ImportError:
    PYPINYIN_OK = False

try:
    import jieba
    jieba.setLogLevel(60)   # silence jieba's startup messages
    JIEBA_OK = True
except ImportError:
    JIEBA_OK = False

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR        = Path(__file__).parent
IMAGE_DIR         = SCRIPT_DIR / "output_images"
ENGLISH_IMAGE_DIR = SCRIPT_DIR / "english_images"
HSK1_CSV          = SCRIPT_DIR / "hsk1.csv"
HSK2_CSV          = SCRIPT_DIR / "hsk2.csv"

# ── Layout constants (Visual Paragraph) ──────────────────────────────────────

CHAR_GAP_PX  = 2    # gap between characters in the same word (nearly touching)
WORD_GAP_PX  = 70   # gap between words (very clearly larger)
LINE_GAP_PX  = 40   # vertical gap between rows
PADDING_PX   = 30   # canvas edge padding
WORD_BG      = (225, 235, 255)   # light blue tint behind each word group
WORD_BORDER  = (180, 200, 240)   # blue-grey outline around each word group
BG_COLOUR    = (255, 255, 255)

# Tile pixel sizes for Small / Medium / Large in the paragraph view
PARA_TILE_PX = {"Small": 120, "Medium": 200, "Large": 320}

# How many tiles per row for each size (keeps output image a reasonable width)
PARA_COLS    = {"Small": 10, "Medium": 7, "Large": 4}

# ── Tone colour system ────────────────────────────────────────────────────────
# Tone border is added around every syllable image tile.
# Tone 1 = yellow, 2 = orange, 3 = green, 4 = red, 0 = neutral gray

TONE_COLOURS = {
    1: (255, 210, 0),    # yellow
    2: (255, 140, 0),    # orange
    3: (50,  168, 82),   # green
    4: (210, 30,  45),   # red
    0: (160, 160, 160),  # gray  (neutral / 5th tone)
}
BORDER_PX = 10   # thickness of the tone border in pixels

# Characters that carry tone marks, mapped to tone number
_TONE_CHARS = {
    1: set("āēīōūǖĀĒĪŌŪǕ"),
    2: set("áéíóúǘÁÉÍÓÚǗ"),
    3: set("ǎěǐǒǔǚǍĚǏǑǓǙ"),
    4: set("àèìòùǜÀÈÌÒÙǛ"),
}

def tone_from_diacritic(syllable: str) -> int:
    """Detect tone number (1–4, or 0 for neutral) from a diacritic-marked syllable."""
    for tone, chars in _TONE_CHARS.items():
        if any(c in chars for c in syllable):
            return tone
    return 0

def tone_from_tone3(syllable_t3: str) -> int:
    """Detect tone number from a TONE3-style string e.g. 'ping2' → 2."""
    if syllable_t3 and syllable_t3[-1].isdigit():
        t = int(syllable_t3[-1])
        return t if 1 <= t <= 4 else 0
    return 0

def chinese_tts(text: str, autoplay: bool) -> None:
    """
    Render a 🔊 button that speaks *text* in Chinese using the browser's
    built-in Web Speech API (no packages or API keys needed).

    When autoplay=True the audio fires automatically ~400 ms after the
    component mounts — long enough for the browser to load voices.
    The button is always visible so the user can replay manually.
    """
    auto_js = "setTimeout(speak, 400);" if autoplay else ""
    components.html(
        f"""
        <div style="text-align:center; margin:4px 0">
          <button onclick="speak()" title="Read aloud" style="
              padding: 6px 22px; font-size: 17px; border-radius: 20px;
              border: 1px solid #ccc; background: #fafafa;
              cursor: pointer; color: #333; line-height:1.4">
            🔊
          </button>
        </div>
        <script>
        function speak() {{
          window.speechSynthesis.cancel();
          var u = new SpeechSynthesisUtterance({json.dumps(text)});
          u.lang  = 'zh-CN';
          u.rate  = 0.85;
          u.pitch = 1.0;
          window.speechSynthesis.speak(u);
        }}
        {auto_js}
        </script>
        """,
        height=52,
        scrolling=False,
    )


def pil_to_b64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data-URI for inline HTML."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()



def add_tone_border(img: Image.Image, tone: int) -> Image.Image:
    """Return a copy of *img* surrounded by a solid-colour tone border."""
    colour = TONE_COLOURS.get(tone, TONE_COLOURS[0])
    w, h   = img.size
    canvas = Image.new("RGB", (w + 2 * BORDER_PX, h + 2 * BORDER_PX), colour)
    canvas.paste(img, (BORDER_PX, BORDER_PX))
    return canvas

# ── Utilities ─────────────────────────────────────────────────────────────────

def to_slug(s: str) -> str:
    """Strip tone marks / diacritics and make filename-safe."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Also strip tone numbers (e.g. ping2 → ping)
    ascii_str = re.sub(r"\d", "", ascii_str)
    return re.sub(r"[^\w]", "_", ascii_str).strip("_").lower()


def image_path(syllable: str) -> Path | None:
    slug = to_slug(syllable)
    for ext in (".jpg", ".png"):
        p = IMAGE_DIR / f"{slug}{ext}"
        if p.exists():
            return p
    return None


def placeholder_image(label: str, size: int = 200) -> Image.Image:
    img = Image.new("RGB", (size, size), (220, 220, 220))
    d = ImageDraw.Draw(img)
    d.text((size // 2, size // 2), label, fill=(80, 80, 80), anchor="mm")
    return img


def build_vocab_lookup(vocab_df: pd.DataFrame) -> dict[str, str]:
    """
    Build a {chinese_word: english} dict from the HSK vocab DataFrame.
    Handles slash-separated variants like '爸爸 / 爸' by using the first form.
    """
    lookup: dict[str, str] = {}
    for _, row in vocab_df.iterrows():
        chinese = str(row.get("chinese", "")).split("/")[0].strip()
        english = str(row.get("english", "")).strip()
        if chinese and english and chinese != "nan":
            lookup[chinese] = english
    return lookup


def assign_english_labels(chars: list[str], lookup: dict[str, str]) -> list[str]:
    """
    Greedily match a list of Chinese characters against the vocab lookup,
    longest match first (up to 4 chars). Returns one English label per
    character position; only the first character of a multi-char match
    gets the label — the rest are empty strings.
    """
    labels = [""] * len(chars)
    i = 0
    while i < len(chars):
        matched = False
        for length in range(min(4, len(chars) - i), 0, -1):
            word = "".join(chars[i : i + length])
            if word in lookup:
                labels[i] = lookup[word]
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return labels


def get_font(size: int):
    """Return a PIL font at the requested size, with a safe fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def split_pinyin(pinyin_str: str) -> list[str]:
    """
    Split a CSV pinyin string into individual syllables.
    Handles diacritics, slash variants ('bà ba / bà'), and multi-syllable words.
    """
    pinyin_str = pinyin_str.split("/")[0].strip()
    parts = pinyin_str.split()
    result = []
    for p in parts:
        p = re.sub(r"[^a-zA-ZüÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]", "", p)
        if p:
            result.append(p.lower())
    return result


# ── Vocabulary loading ────────────────────────────────────────────────────────

@st.cache_data
def load_vocab() -> pd.DataFrame:
    dfs = []
    for path, level in [(HSK1_CSV, "HSK 1"), (HSK2_CSV, "HSK 2")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        # Normalise column names (hsk1 = 'hanzi', hsk2 = 'chinese')
        rename = {}
        for col in df.columns:
            if col in ("hanzi", "chinese"):
                rename[col] = "chinese"
        df = df.rename(columns=rename)
        if "chinese" not in df.columns:
            continue
        df = df[["chinese", "pinyin", "english"]].dropna(subset=["chinese", "pinyin"])
        df["level"] = level
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["chinese", "pinyin", "english", "level"])
    return pd.concat(dfs, ignore_index=True)


@st.cache_data
def load_english_lookup() -> dict[str, dict]:
    """
    Build {slug: {chinese, pinyin, display}} from HSK 1 plus any entries
    in english_words_todo.csv that have a 'your_word' filled in.

    'display' is the human-readable English word or phrase used as the
    multiselect label and as the basis for the image filename.
    """
    lookup: dict[str, dict] = {}

    def _add(chinese: str, pinyin_raw: str, display: str) -> None:
        slug = to_slug(display)
        if not slug or slug in lookup:
            return
        if PYPINYIN_OK:
            chars = "".join(c for c in chinese if "一" <= c <= "鿿")
            py = to_pinyin(chars, style=Style.TONE, heteronym=False)
            pinyin_clean = " ".join(p[0] for p in py if p and p[0])
        else:
            pinyin_clean = pinyin_raw.split("/")[0].strip()
        lookup[slug] = {
            "chinese": chinese.split("/")[0].strip(),
            "pinyin":  pinyin_clean,
            "display": display,
        }

    # ── HSK 1 CSV (all entries) ───────────────────────────────────────────────
    if HSK1_CSV.exists():
        df = pd.read_csv(HSK1_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        hanzi_col = "hanzi" if "hanzi" in df.columns else "chinese"
        for _, row in df.iterrows():
            chinese = str(row.get(hanzi_col, "")).strip()
            pinyin  = str(row.get("pinyin",  "")).strip()
            english = str(row.get("english", "")).strip()
            if not chinese or not english or english == "nan":
                continue
            display = english.split("/")[0].strip()
            if display:
                _add(chinese, pinyin, display)

    # ── User-curated todo CSV (your_word column, any word or phrase) ──────────
    todo_csv = SCRIPT_DIR / "english_words_todo.csv"
    if todo_csv.exists():
        import csv as _csv
        with open(todo_csv, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                chinese          = (row.get("chinese")          or "").strip()
                pinyin           = (row.get("pinyin")           or "").strip()
                your_word        = (row.get("your_word")        or "").strip()
                original_english = (row.get("original_english") or "").strip()
                # Fall back to original English if no override supplied
                display = your_word or original_english.split("/")[0].strip()
                if chinese and display:
                    _add(chinese, pinyin, display)

    return lookup


# ── Visual Paragraph builder ──────────────────────────────────────────────────

def _make_tile(char: str, thumb_px: int) -> dict | None:
    """Convert a single Chinese character into a tile dict {img, syl, tone}."""
    if not ("一" <= char <= "鿿"):
        return None
    py = to_pinyin(char, style=Style.TONE3, heteronym=False)
    syl_t3 = py[0][0].lower() if py and py[0] else ""
    tone   = tone_from_tone3(syl_t3)
    syl    = re.sub(r"\d", "", syl_t3)
    p      = image_path(syl)
    base   = Image.open(p).resize((thumb_px, thumb_px), Image.LANCZOS) if p \
             else placeholder_image(syl, size=thumb_px)
    return {"img": add_tone_border(base, tone), "syl": syl, "tone": tone}


def build_paragraph_image(
    text: str,
    thumb_px: int = 200,
    images_per_row: int = 7,
    show_english: bool = False,
    vocab_lookup: dict | None = None,
) -> Image.Image | None:
    """
    Convert Chinese text into a stitched image of mnemonic syllable tiles.

    Words are segmented with jieba so tiles are grouped by word with a
    visible gap between words. The English translation (when enabled) is
    centred under the full width of each multi-character word.
    """
    if not PYPINYIN_OK:
        return None

    # ── Segment into words ────────────────────────────────────────────────
    if JIEBA_OK:
        raw_words = list(jieba.cut(text, cut_all=False))
    else:
        # Fallback: treat every character as its own word
        raw_words = list(text)

    # ── Build word-level tokens ───────────────────────────────────────────
    # Each token is either:
    #   {"type": "word",  "tiles": [...], "english": str}
    #   {"type": "break"}   ← whitespace / newline between words
    word_tokens: list[dict] = []

    for word in raw_words:
        chinese_chars = [c for c in word if "一" <= c <= "鿿"]

        if chinese_chars:
            tiles = [t for c in chinese_chars for t in [_make_tile(c, thumb_px)] if t]
            english = ""
            if show_english and vocab_lookup:
                # 1. Try the whole word (e.g. 高尔夫球 → "golf")
                english = vocab_lookup.get(word, "")
                # 2. Fall back: look up each character and join meanings
                if not english:
                    char_meanings = [
                        vocab_lookup.get(c, "") for c in chinese_chars
                    ]
                    char_meanings = [m.split("/")[0].split(";")[0].strip()
                                     for m in char_meanings if m]
                    if char_meanings:
                        english = " · ".join(char_meanings)
            word_tokens.append({"type": "word", "tiles": tiles, "english": english})

        elif any(c in " \n\t" for c in word):
            word_tokens.append({"type": "break"})
        # Punctuation: skip silently

    # ── Layout words into rows ────────────────────────────────────────────
    # A word wraps to the next row if it doesn't fit.
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_tiles = 0

    for wt in word_tokens:
        if wt["type"] == "break":
            if current_row:
                current_row.append(wt)
        else:
            n = len(wt["tiles"])
            # Wrap row if this word won't fit (unless the row is empty)
            if current_tiles + n > images_per_row and current_row:
                rows.append(current_row)
                current_row = []
                current_tiles = 0
            current_row.append(wt)
            current_tiles += n

    if current_row:
        rows.append(current_row)

    if not rows:
        return None

    # ── Canvas sizing ─────────────────────────────────────────────────────
    tile_px   = thumb_px + 2 * BORDER_PX
    font_size = max(11, thumb_px // 13)
    text_h    = (font_size + 8) if show_english else 0
    font      = get_font(font_size) if show_english else None

    # Canvas width: enough for images_per_row tiles with word gaps between them
    canvas_w = images_per_row * tile_px + (images_per_row - 1) * WORD_GAP_PX + PADDING_PX * 2
    row_h    = tile_px + text_h
    n_rows   = len(rows)
    canvas_h = n_rows * row_h + (n_rows - 1) * LINE_GAP_PX + PADDING_PX * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOUR)
    draw   = ImageDraw.Draw(canvas)

    # ── Draw ──────────────────────────────────────────────────────────────
    y = PADDING_PX
    for row in rows:
        x = PADDING_PX
        for wt in row:
            if wt["type"] == "break":
                x += WORD_GAP_PX
                continue

            tiles    = wt["tiles"]
            n        = len(tiles)
            word_w   = n * tile_px + (n - 1) * CHAR_GAP_PX
            word_x   = x   # remember where this word starts

            # Draw a filled + outlined box behind the whole word group
            pad = 5
            box = [word_x - pad, y - pad, word_x + word_w + pad, y + tile_px + pad]
            draw.rectangle(box, fill=WORD_BG, outline=WORD_BORDER, width=2)

            # Draw each tile tightly within the word
            for tile in tiles:
                canvas.paste(tile["img"], (x, y))
                x += tile_px + CHAR_GAP_PX

            # English label centred under the full word width
            if show_english and font and wt.get("english"):
                label = wt["english"].replace("\n", " ")
                while len(label) > 3 and draw.textlength(label, font=font) > word_w:
                    label = label[:-2] + "…"
                text_w = draw.textlength(label, font=font)
                text_x = word_x + (word_w - text_w) // 2
                text_y = y + tile_px + 3
                draw.text((text_x, text_y), label, fill=(60, 60, 60), font=font)

            # Advance by word gap (replacing the last char gap)
            x += WORD_GAP_PX - CHAR_GAP_PX

        y += row_h + LINE_GAP_PX

    return canvas


def show_paragraph_as_images(
    text: str,
    thumb_px: int = 200,
    images_per_row: int = 7,
    show_english: bool = False,
    vocab_lookup: dict | None = None,
) -> bool:
    """
    Display paragraph tiles as individual st.image() calls so each one gets
    Streamlit's native click-to-expand zoom button.

    Words are grouped visually using narrow spacer columns between word groups.
    Returns True if at least one tile was rendered.
    """
    if not PYPINYIN_OK:
        return False

    # ── Segment into words ────────────────────────────────────────────────────
    raw_words = list(jieba.cut(text, cut_all=False)) if JIEBA_OK else list(text)

    word_tokens: list[dict] = []
    for word in raw_words:
        chinese_chars = [c for c in word if "一" <= c <= "鿿"]
        if not chinese_chars:
            if any(c in " \n\t" for c in word):
                word_tokens.append({"type": "break"})
            continue
        tiles = [t for c in chinese_chars for t in [_make_tile(c, thumb_px)] if t]
        english = ""
        if show_english and vocab_lookup:
            english = vocab_lookup.get(word, "")
            if not english:
                char_meanings = [vocab_lookup.get(c, "") for c in chinese_chars]
                char_meanings = [m.split("/")[0].split(";")[0].strip()
                                 for m in char_meanings if m]
                if char_meanings:
                    english = " · ".join(char_meanings)
        if tiles:
            word_tokens.append({"type": "word", "tiles": tiles, "english": english})

    # ── Layout into rows ──────────────────────────────────────────────────────
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_tiles = 0
    for wt in word_tokens:
        if wt["type"] == "break":
            if current_row:
                rows.append(current_row)
                current_row = []
                current_tiles = 0
        else:
            n = len(wt["tiles"])
            if current_tiles + n > images_per_row and current_row:
                rows.append(current_row)
                current_row = []
                current_tiles = 0
            current_row.append(wt)
            current_tiles += n
    if current_row:
        rows.append(current_row)

    if not rows:
        return False

    # ── Render each row ───────────────────────────────────────────────────────
    for row in rows:
        word_groups = [wt for wt in row if wt["type"] == "word"]
        if not word_groups:
            continue

        # Column spec: each tile = width 1, narrow gap (0.25) between word groups
        col_spec: list[float] = []
        for i, wt in enumerate(word_groups):
            if i > 0:
                col_spec.append(0.25)   # word-group separator
            col_spec.extend([1.0] * len(wt["tiles"]))

        all_cols = st.columns(col_spec)

        # Place images and record which columns belong to each word group
        col_idx = 0
        word_col_ranges: list[tuple[int, int]] = []
        for i, wt in enumerate(word_groups):
            if i > 0:
                col_idx += 1            # skip separator column
            word_start = col_idx
            for tile in wt["tiles"]:
                with all_cols[col_idx]:
                    st.image(tile["img"], use_container_width=True)
                col_idx += 1
            word_col_ranges.append((word_start, col_idx - 1))

        # English labels in a second band of the same-width columns
        if show_english and any(wt.get("english") for wt in word_groups):
            label_cols = st.columns(col_spec)
            for (start, end), wt in zip(word_col_ranges, word_groups):
                if wt.get("english"):
                    mid = (start + end) // 2
                    with label_cols[mid]:
                        st.caption(wt["english"][:30])

        st.write("")   # small vertical gap between rows

    return True


# ── Flashcard state helpers ───────────────────────────────────────────────────

def reset_flashcards():
    for k in ("fc_indices", "fc_pos", "fc_revealed", "fc_known", "fc_learning", "fc_level_key"):
        st.session_state.pop(k, None)


def init_flashcards(df: pd.DataFrame):
    """
    Initialise the flashcard deck using *positional* indices (0…n-1) so that
    .iloc always works regardless of the DataFrame's label index.
    Automatically resets when the level selection changes.
    """
    # A hashable key representing which levels are currently loaded
    level_key = tuple(sorted(df["level"].unique()))

    if (
        "fc_indices" not in st.session_state
        or st.session_state.get("fc_level_key") != level_key
    ):
        indices = list(range(len(df)))
        random.shuffle(indices)
        st.session_state.fc_indices   = indices
        st.session_state.fc_pos       = 0
        st.session_state.fc_revealed  = False
        st.session_state.fc_known     = set()
        st.session_state.fc_learning  = set()
        st.session_state.fc_level_key = level_key


# ── App layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pinyin Visual Learner",
    page_icon="🀄",
    layout="centered",
)

st.title("🀄 Pinyin Visual Learner")

if not PYPINYIN_OK:
    st.error(
        "**pypinyin** is not installed — the Visual Paragraph tab won't work.  \n"
        "Fix it by running:  \n```\npip install pypinyin\n```"
    )

vocab = load_vocab()

tab_study, tab_para, tab_words = st.tabs(["📚 Study", "🖼️ Visual Paragraph", "🔤 Word Lookup"])


# ── Sidebar (visible in both tabs) ───────────────────────────────────────────

# How many equal columns to split the flashcard image row into per size.
# More columns = narrower images.
COLS_FOR_SIZE = {"Small": 5, "Medium": 3, "Large": 2}

with st.sidebar:
    st.header("Settings")

    levels = st.multiselect(
        "HSK Level",
        options=["HSK 1", "HSK 2"],
        default=["HSK 1"],
    )

    filtered = vocab[vocab["level"].isin(levels)].copy() if levels else pd.DataFrame()
    st.caption(f"{len(filtered)} words loaded")

    img_size = st.radio("Image size", ["Small", "Medium", "Large"], index=1)

    auto_read = st.toggle("🔊 Read aloud automatically", value=True)

    if st.button("🔀 Shuffle & Restart", use_container_width=True):
        reset_flashcards()
        st.rerun()

    # Tone colour legend
    st.divider()
    st.markdown("**Tone colours**")
    for label, colour in [
        ("Tone 1 — flat",    TONE_COLOURS[1]),
        ("Tone 2 — rising",  TONE_COLOURS[2]),
        ("Tone 3 — dip",     TONE_COLOURS[3]),
        ("Tone 4 — falling", TONE_COLOURS[4]),
        ("Neutral",          TONE_COLOURS[0]),
    ]:
        hex_c = "#{:02x}{:02x}{:02x}".format(*colour)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<div style='width:16px;height:16px;border-radius:3px;"
            f"background:{hex_c};flex-shrink:0'></div>{label}</div>",
            unsafe_allow_html=True,
        )

    # Diagnostic: show where the app is looking for images
    st.divider()
    img_count = sum(1 for ext in ("*.jpg", "*.png") for _ in IMAGE_DIR.glob(ext)) if IMAGE_DIR.exists() else 0
    if img_count:
        st.caption(f"📁 {img_count} images found")
    else:
        st.warning(f"⚠️ No images found in:\n`{IMAGE_DIR}`")

    # Progress summary
    if "fc_known" in st.session_state:
        st.divider()
        st.metric("✅ Known",          len(st.session_state.fc_known))
        st.metric("🔁 Still learning", len(st.session_state.fc_learning))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FLASHCARDS
# ═══════════════════════════════════════════════════════════════════════════════

def show_syllable_images(syllables: list[str], size: str) -> None:
    """
    Display syllable images in a centred row using st.image().

    The number of grid columns is driven by `size` (Small/Medium/Large).
    When there are fewer syllables than the target column count, spacer
    columns are added on both sides so images appear centred.
    """
    n = len(syllables)
    if n == 0:
        return

    target_cols = COLS_FOR_SIZE[size]
    # Always use at least as many columns as there are syllables
    num_cols = max(n, target_cols)
    all_cols = st.columns(num_cols)

    # Centre n images inside num_cols by offsetting from the left
    offset = (num_cols - n) // 2
    img_cols = all_cols[offset : offset + n]

    for syl, col in zip(syllables, img_cols):
        tone = tone_from_diacritic(syl)
        p    = image_path(syl)
        with col:
            if p:
                base     = Image.open(p)
                bordered = add_tone_border(base, tone)
                st.image(bordered, use_container_width=True)
            else:
                hex_col = "#{:02x}{:02x}{:02x}".format(
                    *TONE_COLOURS.get(tone, TONE_COLOURS[0])
                )
                st.markdown(
                    f"<div style='text-align:center;padding:20px;"
                    f"border:6px solid {hex_col};border-radius:8px;"
                    f"font-size:1.1em;font-weight:bold'>{syl}<br>"
                    f"<span style='font-size:0.7em;font-weight:normal'>"
                    f"image not found</span></div>",
                    unsafe_allow_html=True,
                )


with tab_study:
    if filtered.empty:
        st.info("Select at least one HSK level in the sidebar to begin.")
    else:
        init_flashcards(filtered)

        pos     = st.session_state.fc_pos
        indices = st.session_state.fc_indices
        total   = len(indices)

        if pos >= total:
            # ── Finished ──────────────────────────────────────────────────
            st.balloons()
            st.success(f"🎉 You've been through all {total} cards!")
            col_a, col_b = st.columns(2)
            col_a.metric("✅ Known",          len(st.session_state.fc_known))
            col_b.metric("🔁 Still learning", len(st.session_state.fc_learning))
            if st.button("▶ Go again", use_container_width=True):
                reset_flashcards()
                st.rerun()

        else:
            row = filtered.iloc[indices[pos]]

            # Progress bar
            st.progress(pos / total, text=f"Card {pos + 1} / {total}")

            # ── Chinese word — large and centred ──────────────────────────
            chinese_text = str(row["chinese"])
            st.markdown(
                f"<h1 style='text-align:center; font-size:80px; "
                f"letter-spacing:0.1em; margin:0.3em 0'>{chinese_text}</h1>",
                unsafe_allow_html=True,
            )

            # 🔊 Audio — auto-plays on new card (if toggle on), manual button always shown.
            # Autoplay is only active before reveal so it doesn't re-fire on "Reveal" click.
            chinese_tts(chinese_text, autoplay=auto_read and not st.session_state.fc_revealed)

            # ── Syllable images ────────────────────────────────────────────
            # Derive syllables from the Chinese characters directly so that
            # multi-syllable words (e.g. 完全 → wán + quán) are always split
            # correctly regardless of how the CSV pinyin field is formatted.
            if PYPINYIN_OK:
                chars_only = "".join(c for c in chinese_text if "一" <= c <= "鿿")
                py_list    = to_pinyin(chars_only, style=Style.TONE, heteronym=False)
                syllables  = [py[0].lower() for py in py_list if py and py[0]]
            else:
                syllables = split_pinyin(str(row["pinyin"]))
            show_syllable_images(syllables, img_size)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Reveal / Rate ──────────────────────────────────────────────
            if not st.session_state.fc_revealed:
                if st.button("👁️  Reveal meaning", use_container_width=True):
                    st.session_state.fc_revealed = True
                    st.rerun()
            else:
                st.markdown(
                    f"<h3 style='text-align:center;color:#555'>{row['pinyin']}</h3>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<h2 style='text-align:center'>{row['english']}</h2>",
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                btn_l, btn_r = st.columns(2)

                if btn_l.button("✅  Know it", use_container_width=True):
                    st.session_state.fc_known.add(pos)
                    st.session_state.fc_pos += 1
                    st.session_state.fc_revealed = False
                    st.rerun()

                if btn_r.button("🔁  Still learning", use_container_width=True):
                    st.session_state.fc_learning.add(pos)
                    st.session_state.fc_pos += 1
                    st.session_state.fc_revealed = False
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUAL PARAGRAPH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_para:
    st.subheader("Visual Paragraph Generator")
    st.write(
        "Paste any Chinese text below and it will be rendered as a paragraph "
        "using your mnemonic syllable images — one tile per character."
    )

    sample = "我喜欢学习中文。每天练习很重要。"
    chinese_input = st.text_area(
        "Chinese text",
        height=130,
        placeholder=f"e.g.  {sample}",
    )

    show_english = st.toggle("Show English translations below each image", value=True)

    col_gen, col_sample = st.columns([2, 1])
    generate_clicked = col_gen.button("🎨  Generate visual paragraph", use_container_width=True)
    sample_clicked   = col_sample.button("Try a sample sentence", use_container_width=True)

    input_text = sample if sample_clicked else chinese_input.strip()

    if (generate_clicked or sample_clicked) and input_text:
        if not PYPINYIN_OK:
            st.error("pypinyin is required. Run `pip install pypinyin` then restart the app.")
        else:
            # ── Debug: show jieba segmentation ────────────────────────────
            if JIEBA_OK:
                words = [w for w in jieba.cut(input_text, cut_all=False)
                         if any("一" <= c <= "鿿" for c in w)]
                with st.expander("🔍 Jieba word segmentation (debug)"):
                    st.write(" | ".join(words))
            else:
                st.warning("jieba not available — install it with `pip install jieba`")

            v_lookup = build_vocab_lookup(vocab) if show_english else {}

            displayed = show_paragraph_as_images(
                input_text,
                thumb_px=PARA_TILE_PX[img_size],
                images_per_row=PARA_COLS[img_size],
                show_english=show_english,
                vocab_lookup=v_lookup,
            )

            if not displayed:
                st.warning("No recognisable Chinese characters found in the input.")
            else:
                # Download button — builds the flat PIL image for export
                with st.spinner("Preparing download image…"):
                    para_img = build_paragraph_image(
                        input_text,
                        thumb_px=PARA_TILE_PX[img_size],
                        images_per_row=PARA_COLS[img_size],
                        show_english=show_english,
                        vocab_lookup=v_lookup,
                    )
                if para_img:
                    buf = io.BytesIO()
                    para_img.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️  Download as PNG",
                        data=buf.getvalue(),
                        file_name="visual_paragraph.png",
                        mime="image/png",
                        use_container_width=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WORD LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

# ── Pre-built Chinglish passage library ───────────────────────────────────────
# Each passage is written in Chinese grammar word order (no articles,
# time-words first, serial verbs) using only HSK 1 & 2 vocabulary.
# Grey tiles will appear for any words not yet in the image library.

_PASSAGE_NONE = "— pick a story —"
PASSAGE_LIBRARY = [
    {
        "title": "My Morning Routine",
        "level": "HSK 1",
        "text": (
            "Today morning I eat breakfast.\n"
            "I drink water and tea.\n"
            "After I go school.\n"
            "Teacher teach Chinese.\n"
            "I like study Chinese.\n"
            "Afternoon I very tired.\n"
            "I want rest."
        ),
    },
    {
        "title": "Shopping Day",
        "level": "HSK 1",
        "text": (
            "Tomorrow I want go shop.\n"
            "Shop sell clothes and book.\n"
            "I buy red clothes.\n"
            "Clothes very beautiful.\n"
            "I also buy book.\n"
            "Book very interesting.\n"
            "I happy go home."
        ),
    },
    {
        "title": "At the Hospital",
        "level": "HSK 1",
        "text": (
            "Yesterday I not comfortable.\n"
            "I go hospital see doctor.\n"
            "Doctor say I need rest.\n"
            "Doctor say drink more water.\n"
            "I buy medicine.\n"
            "Now I better.\n"
            "Tomorrow I go school."
        ),
    },
    {
        "title": "Cold Weather",
        "level": "HSK 1–2",
        "text": (
            "Today weather very cold.\n"
            "Outside have snow.\n"
            "I wear overcoat go outside.\n"
            "Afternoon rain stop.\n"
            "Weather become warm.\n"
            "Evening I at home watch television.\n"
            "Today I not go outside."
        ),
    },
    {
        "title": "My Family",
        "level": "HSK 1",
        "text": (
            "I family have dad mom and I.\n"
            "Dad go office work.\n"
            "Mom go shop buy food.\n"
            "Evening we together eat dinner.\n"
            "I help mom wash bowl.\n"
            "After we watch television.\n"
            "We family very happy."
        ),
    },
    {
        "title": "Learning Chinese",
        "level": "HSK 1–2",
        "text": (
            "I study Chinese every day.\n"
            "Chinese very interesting.\n"
            "Morning I read book.\n"
            "I listen teacher speak Chinese.\n"
            "I write Chinese character.\n"
            "Sometimes I speak Chinese with friend.\n"
            "My Chinese slowly become good."
        ),
    },
    {
        "title": "Weekend Fun",
        "level": "HSK 1–2",
        "text": (
            "Saturday I not go school.\n"
            "Morning I run park.\n"
            "Park very beautiful.\n"
            "Afternoon friend come my home.\n"
            "We together watch movie.\n"
            "Evening we go restaurant eat dinner.\n"
            "Weekend very happy."
        ),
    },
    {
        "title": "At the Restaurant",
        "level": "HSK 1–2",
        "text": (
            "Today evening I go restaurant eat dinner.\n"
            "Restaurant sell noodles and dumplings.\n"
            "I want eat noodles.\n"
            "Friend want eat dumplings.\n"
            "Food very delicious.\n"
            "We drink tea.\n"
            "Bill not expensive.\n"
            "We very happy."
        ),
    },
    {
        "title": "Travel to Beijing",
        "level": "HSK 1–2",
        "text": (
            "Next month I go Beijing travel.\n"
            "I take airplane go.\n"
            "Airplane very fast.\n"
            "Beijing very big and beautiful.\n"
            "I want see many place.\n"
            "I also want buy souvenir.\n"
            "Travel very interesting."
        ),
    },
    {
        "title": "My Good Friend",
        "level": "HSK 1–2",
        "text": (
            "I have good friend.\n"
            "He very smart and happy.\n"
            "We together study Chinese.\n"
            "Sometimes we go cinema see movie.\n"
            "Sometimes we go park run.\n"
            "Friend very important.\n"
            "I very happy have this friend."
        ),
    },
    {
        "title": "Dad Goes to Work",
        "level": "HSK 1–2",
        "text": (
            "Every day morning dad wake up early.\n"
            "He eat breakfast drink coffee.\n"
            "After he drive car go office.\n"
            "Office very busy.\n"
            "Afternoon he have meeting.\n"
            "Evening dad come home very tired.\n"
            "Mom cook dinner wait him."
        ),
    },
    {
        "title": "The Four Seasons",
        "level": "HSK 1–2",
        "text": (
            "Spring weather warm.\n"
            "Outside flower bloom.\n"
            "Summer weather very hot.\n"
            "I want drink cold water.\n"
            "Autumn weather very good.\n"
            "I like go outside walk.\n"
            "Winter weather cold.\n"
            "I wear overcoat."
        ),
    },
    {
        "title": "A Rainy Day",
        "level": "HSK 1–2",
        "text": (
            "Today morning weather cloudy.\n"
            "Afternoon start rain.\n"
            "I not bring umbrella.\n"
            "I go convenience store buy umbrella.\n"
            "Rain very big.\n"
            "I wait inside shop.\n"
            "Later rain stop.\n"
            "I go home."
        ),
    },
    {
        "title": "At the Supermarket",
        "level": "HSK 2",
        "text": (
            "Today mom want go supermarket.\n"
            "I together go.\n"
            "Supermarket sell many thing.\n"
            "Mom buy vegetable and meat.\n"
            "I want buy bread and milk.\n"
            "Supermarket very convenient.\n"
            "We buy finish go home.\n"
            "Mom cook very delicious food."
        ),
    },
]


def build_english_passage_image(
    token_stream: list[dict],
    eng_lookup: dict,
    thumb_px: int = 180,
    cols_per_row: int = 5,
) -> Image.Image | None:
    """
    Stitch English mnemonic tiles into a single downloadable PIL image.
    Available words use their mnemonic JPEG; missing words get a grey box.
    """
    word_tokens = [t for t in token_stream if t["type"] == "word"]
    if not word_tokens:
        return None

    # Re-layout into rows (same logic as the live display)
    rows: list[list[dict]] = []
    current: list[dict] = []
    for tok in token_stream:
        if tok["type"] == "newline":
            if current:
                rows.append(current)
                current = []
        else:
            if len(current) >= cols_per_row:
                rows.append(current)
                current = []
            current.append(tok)
    if current:
        rows.append(current)

    GAP_X    = 10
    GAP_Y    = 14
    LABEL_H  = max(18, thumb_px // 10)
    PAD      = 20
    font     = get_font(max(13, thumb_px // 12))
    chin_font = get_font(max(18, thumb_px // 9))

    row_h    = thumb_px + LABEL_H + 4
    canvas_w = cols_per_row * thumb_px + (cols_per_row - 1) * GAP_X + PAD * 2
    canvas_h = len(rows) * row_h + (len(rows) - 1) * GAP_Y + PAD * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    y = PAD
    for row in rows:
        x = PAD
        for tok in row:
            img_p = ENGLISH_IMAGE_DIR / f"{tok['slug']}.jpg"
            if tok["found"] and img_p.exists():
                tile = Image.open(img_p).resize((thumb_px, thumb_px), Image.LANCZOS)
            else:
                tile = Image.new("RGB", (thumb_px, thumb_px), (220, 220, 220))
                td   = ImageDraw.Draw(tile)
                tw   = td.textlength(tok["text"], font=font)
                td.text(
                    ((thumb_px - tw) / 2, thumb_px / 2 - LABEL_H),
                    tok["text"], fill=(120, 120, 120), font=font,
                )

            canvas.paste(tile, (x, y))

            # Word label centred below tile
            label = tok["text"]
            lw = draw.textlength(label, font=font)
            draw.text(
                (x + (thumb_px - lw) / 2, y + thumb_px + 3),
                label, fill=(140, 140, 140), font=font,
            )
            x += thumb_px + GAP_X
        y += row_h + GAP_Y

    return canvas


# Chinglish sentences: English words in Chinese word order.
# Each 'words' list uses the exact English display names from HSK 1.
_SENTENCE_NONE = "— choose a sample sentence —"
CURATED_SENTENCES = [
    {"label": "I morning eat breakfast",             "words": ["morning", "eat", "breakfast"]},
    {"label": "She afternoon drink tea rest",        "words": ["afternoon", "drink", "tea", "rest"]},
    {"label": "I evening go to bed",                 "words": ["evening", "go to bed"]},
    {"label": "He today very busy",                  "words": ["today", "busy"]},
    {"label": "I want drink water",                  "words": ["want", "drink", "water"]},
    {"label": "She buy clothes",                     "words": ["buy", "clothes"]},
    {"label": "I go bookstore buy book",             "words": ["bookstore", "buy", "book"]},
    {"label": "I like eat bread",                    "words": ["like", "eat", "bread"]},
    {"label": "He tomorrow go Beijing",              "words": ["tomorrow", "go", "Beijing"]},
    {"label": "Teacher ask student",                 "words": ["teacher", "ask", "student"]},
    {"label": "I dad together drink tea",            "words": ["dad", "together", "drink", "tea"]},
    {"label": "Son daughter play ball",              "words": ["son", "daughter", "ball"]},
    {"label": "I birthday eat dinner",               "words": ["birthday", "eat", "dinner"]},
    {"label": "He drive car go airport",             "words": ["drive", "car", "airport"]},
    {"label": "Weather today very cold",             "words": ["weather", "today", "cold"]},
    {"label": "I go cinema see movie",               "words": ["cinema", "movie"]},
    {"label": "I classmate go school together",      "words": ["classmate", "school", "together"]},
    {"label": "Doctor come hospital",                "words": ["doctor", "come", "hospital"]},
    {"label": "I at home watch television",          "words": ["at home", "television"]},
    {"label": "I use cell phone call friend",        "words": ["cell phone", "call", "friend"]},
]


# Word categories for random sentence generation.
# Values must match the exact 'display' names produced by load_english_lookup().
_GEN_CATEGORIES: dict[str, list[str]] = {
    "time":   ["today", "tomorrow", "yesterday", "morning", "afternoon", "evening"],
    "person": ["dad", "friend", "teacher", "doctor", "classmate",
               "son", "daughter", "boyfriend", "girlfriend"],
    "verb":   ["eat", "drink", "buy", "drive", "want", "like", "rest", "study", "call"],
    "food":   ["breakfast", "lunch", "dinner", "bread", "tea", "water", "bun", "noodles"],
    "item":   ["book", "clothes", "bag", "ball", "car", "computer", "cell phone", "backpack"],
    "place":  ["school", "hospital", "bookstore", "cinema", "airport",
               "Beijing", "restaurant", "library"],
    "adj":    ["busy", "happy", "cold", "tired", "early", "clean"],
}

# Each template is a list of category names in Chinese grammar word order.
_GEN_TEMPLATES: list[list[str]] = [
    ["time", "verb", "food"],
    ["person", "time", "adj"],
    ["person", "verb", "item"],
    ["time", "verb", "place"],
    ["time", "person", "verb", "place"],
    ["person", "verb", "food", "place"],
    ["time", "person", "verb", "food"],
    ["person", "verb", "item", "place"],
]


def _load_wl_sentence() -> None:
    label = st.session_state.get("wl_sentence_picker", "")
    if not label or label == _SENTENCE_NONE:
        return
    chosen = next((s for s in CURATED_SENTENCES if s["label"] == label), None)
    if not chosen:
        return
    eng_lkp = load_english_lookup()
    avail = {v["display"] for k, v in eng_lkp.items()
             if (ENGLISH_IMAGE_DIR / f"{k}.jpg").exists()}
    st.session_state.wl_selected = [w for w in chosen["words"] if w in avail]


def _generate_wl_sentence(available: dict[str, str]) -> list[str]:
    """Pick a random template and fill each slot from available image words."""
    template = random.choice(_GEN_TEMPLATES)
    result = []
    for cat in template:
        candidates = [w for w in _GEN_CATEGORIES[cat] if w in available]
        if candidates:
            result.append(random.choice(candidates))
    return result


with tab_words:
    st.subheader("Word Lookup")

    eng_lookup = load_english_lookup()

    # Only surface entries that already have a generated image
    available: dict[str, str] = {          # display → slug
        v["display"]: k
        for k, v in eng_lookup.items()
        if (ENGLISH_IMAGE_DIR / f"{k}.jpg").exists()
    }

    # ── Mode toggle ───────────────────────────────────────────────────────────
    wl_mode = st.radio(
        "Mode",
        ["🔍 Word Search", "📖 Passage Reader"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ═════════════════════════════════════════════════════════════════════════
    # MODE A — WORD SEARCH (original multiselect behaviour)
    # ═════════════════════════════════════════════════════════════════════════
    if wl_mode == "🔍 Word Search":
        st.write(
            "Search for English words or phrases — type to filter, then select. "
            "Each entry shows the Chinese mnemonic art for that word or phrase."
        )

        col_sentence, col_generate = st.columns([4, 1])
        with col_sentence:
            st.selectbox(
                "Or load a sample sentence:",
                options=[_SENTENCE_NONE] + [s["label"] for s in CURATED_SENTENCES],
                key="wl_sentence_picker",
                on_change=_load_wl_sentence,
            )
        with col_generate:
            st.write("")   # nudge button down to align with selectbox
            if st.button("🎲 Generate", use_container_width=True):
                words = _generate_wl_sentence(available)
                if words:
                    st.session_state.wl_selected = words

        selected_displays = st.multiselect(
            "Words / phrases",
            options=sorted(available.keys(), key=str.lower),
            placeholder="Type to search — e.g.  sleep,  phone,  play ball…",
            key="wl_selected",
        )

        if selected_displays:
            cols_per_row = COLS_FOR_SIZE[img_size]
            slugs = [available[d] for d in selected_displays]

            for row_start in range(0, len(slugs), cols_per_row):
                row_slugs = slugs[row_start : row_start + cols_per_row]
                padded    = row_slugs + [None] * (cols_per_row - len(row_slugs))
                cols      = st.columns(cols_per_row)

                for col, slug in zip(cols, padded):
                    if slug is None:
                        continue
                    entry = eng_lookup[slug]
                    img_p = ENGLISH_IMAGE_DIR / f"{slug}.jpg"

                    with col:
                        st.image(Image.open(img_p), use_container_width=True)
                        st.markdown(
                            f"<div style='text-align:center;font-size:2em;"
                            f"font-weight:bold;margin-top:4px'>"
                            f"{entry['chinese']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='text-align:center;color:#666;"
                            f"margin-bottom:8px'>{entry['pinyin']}</div>",
                            unsafe_allow_html=True,
                        )

                st.write("")

    # ═════════════════════════════════════════════════════════════════════════
    # MODE B — PASSAGE READER
    # ═════════════════════════════════════════════════════════════════════════
    else:
        st.write(
            "Pick a ready-made Chinglish story **or** type your own passage below. "
            "Each word is shown as its mnemonic image — grey dashed tiles are "
            "words not yet in the library."
        )

        # ── Story library picker ──────────────────────────────────────────────
        story_options = [_PASSAGE_NONE] + [
            f"{p['title']}  ({p['level']})" for p in PASSAGE_LIBRARY
        ]
        chosen_story = st.selectbox(
            "📚  Load a story",
            options=story_options,
            key="wl_story_picker",
        )
        if chosen_story != _PASSAGE_NONE:
            idx = story_options.index(chosen_story) - 1
            st.session_state["wl_passage_rendered"] = PASSAGE_LIBRARY[idx]["text"]

        # ── Free-text area ────────────────────────────────────────────────────
        passage_text = st.text_area(
            "Or type / paste your own passage (write in Chinese word order for best effect):",
            value=st.session_state.get("wl_passage_rendered", ""),
            height=160,
            placeholder=(
                "e.g.  Today morning I eat breakfast.\n"
                "After I go school study Chinese.\n"
                "Afternoon I go shop buy book."
            ),
            key="wl_passage_input",
        )

        col_read, col_clear = st.columns([3, 1])
        read_clicked  = col_read.button("📖  Render passage", use_container_width=True)
        clear_clicked = col_clear.button("✕  Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.pop("wl_passage_rendered", None)
            st.rerun()

        if read_clicked and passage_text.strip():
            st.session_state["wl_passage_rendered"] = passage_text.strip()

        rendered_text = st.session_state.get("wl_passage_rendered", "")
        # Sync: if the user manually edited the text area, use that
        if passage_text.strip() and passage_text.strip() != rendered_text:
            rendered_text = passage_text.strip() if read_clicked else rendered_text

        if rendered_text:
            # ── Tokenise: split into words, keep newlines as row breaks ───────
            # Build a token stream: {"type": "word", "text": str}
            #                    or {"type": "newline"}
            # We try 2-word combinations first (for phrases like "play ball").
            token_stream: list[dict] = []
            for line in rendered_text.splitlines():
                if token_stream:
                    token_stream.append({"type": "newline"})
                raw_words = re.findall(r"[a-zA-Z''-]+", line)
                i = 0
                while i < len(raw_words):
                    # Try a 2-word phrase first (e.g. "play ball", "at home")
                    matched = False
                    if i + 1 < len(raw_words):
                        phrase = f"{raw_words[i]} {raw_words[i + 1]}"
                        slug2  = to_slug(phrase)
                        if (ENGLISH_IMAGE_DIR / f"{slug2}.jpg").exists():
                            token_stream.append({"type": "word", "text": phrase,
                                                 "slug": slug2, "found": True})
                            i += 2
                            matched = True
                    if not matched:
                        word = raw_words[i]
                        slug = to_slug(word)
                        found = (ENGLISH_IMAGE_DIR / f"{slug}.jpg").exists()
                        token_stream.append({"type": "word", "text": word,
                                             "slug": slug, "found": found})
                        i += 1

            # ── Layout into rows (wrap at cols_per_row words) ─────────────────
            cols_per_row = COLS_FOR_SIZE[img_size]
            rows: list[list[dict]] = []
            current_row: list[dict] = []

            for tok in token_stream:
                if tok["type"] == "newline":
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                else:
                    if len(current_row) >= cols_per_row:
                        rows.append(current_row)
                        current_row = []
                    current_row.append(tok)
            if current_row:
                rows.append(current_row)

            # ── Render rows ───────────────────────────────────────────────────
            found_count = sum(1 for t in token_stream
                              if t["type"] == "word" and t["found"])
            total_count = sum(1 for t in token_stream if t["type"] == "word")
            st.caption(
                f"{found_count} / {total_count} words have mnemonic images  "
                f"({'grey' if found_count < total_count else 'all covered ✅'}  "
                f"{'tiles = not yet generated' if found_count < total_count else ''})"
            )
            st.write("")

            for row in rows:
                padded = row + [None] * (cols_per_row - len(row))
                cols   = st.columns(cols_per_row)

                for col, tok in zip(cols, padded):
                    if tok is None:
                        continue
                    with col:
                        if tok["found"]:
                            img_p = ENGLISH_IMAGE_DIR / f"{tok['slug']}.jpg"
                            st.image(Image.open(img_p), use_container_width=True)
                            # Chinese + pinyin below
                            entry = eng_lookup.get(tok["slug"], {})
                            if entry.get("chinese"):
                                st.markdown(
                                    f"<div style='text-align:center;"
                                    f"font-size:1.5em;font-weight:bold;"
                                    f"line-height:1.2;margin-top:2px'>"
                                    f"{entry['chinese']}</div>",
                                    unsafe_allow_html=True,
                                )
                            if entry.get("pinyin"):
                                st.markdown(
                                    f"<div style='text-align:center;"
                                    f"color:#888;font-size:0.82em;"
                                    f"margin-bottom:6px'>{entry['pinyin']}</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            # Grey placeholder — word not in library yet
                            st.markdown(
                                f"<div style='"
                                f"background:#ebebeb;"
                                f"border:2px dashed #ccc;"
                                f"border-radius:8px;"
                                f"padding:18px 6px;"
                                f"text-align:center;"
                                f"color:#999;"
                                f"font-size:0.9em;"
                                f"font-weight:600;"
                                f"min-height:70px;"
                                f"display:flex;"
                                f"align-items:center;"
                                f"justify-content:center;'>"
                                f"{tok['text']}</div>",
                                unsafe_allow_html=True,
                            )

                        # Word label under every cell
                        st.markdown(
                            f"<div style='text-align:center;font-size:0.75em;"
                            f"color:#bbb;margin-top:2px'>{tok['text']}</div>",
                            unsafe_allow_html=True,
                        )

                st.write("")

            # ── Download button ───────────────────────────────────────────────
            st.divider()
            with st.spinner("Building download image…"):
                dl_img = build_english_passage_image(
                    token_stream,
                    eng_lookup,
                    thumb_px=PARA_TILE_PX[img_size],
                    cols_per_row=COLS_FOR_SIZE[img_size],
                )
            if dl_img:
                buf = io.BytesIO()
                dl_img.save(buf, format="PNG")
                st.download_button(
                    label="⬇️  Download passage as PNG",
                    data=buf.getvalue(),
                    file_name="passage.png",
                    mime="image/png",
                    use_container_width=True,
                )
