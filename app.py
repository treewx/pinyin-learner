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

SCRIPT_DIR = Path(__file__).parent
IMAGE_DIR  = SCRIPT_DIR / "output_images"
HSK1_CSV   = SCRIPT_DIR / "hsk1.csv"
HSK2_CSV   = SCRIPT_DIR / "hsk2.csv"

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

tab_study, tab_para = st.tabs(["📚 Study", "🖼️ Visual Paragraph"])


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
    img_count = len(list(IMAGE_DIR.glob("*.png"))) if IMAGE_DIR.exists() else 0
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
