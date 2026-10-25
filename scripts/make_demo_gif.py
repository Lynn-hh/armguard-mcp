"""Render docs/demo.gif: the same abridged session as docs/demo.svg, as an animated GIF.

GitHub renders the SVG, but PyPI, social previews and most chat apps do not animate it. The GIF reuses the
line list of ``make_demo_svg.py`` (every line is shortened from the real output in docs/demo.md), so the two
never drift apart.  Re-run after changing the demo:  python scripts/make_demo_gif.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"

_spec = importlib.util.spec_from_file_location("make_demo_svg", ROOT / "scripts" / "make_demo_svg.py")
_svg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_svg)
LINES, FG, DIM = _svg.LINES, _svg.FG, _svg.DIM

W, PAD, LH, TOP = _svg.W, _svg.PAD, _svg.LH, _svg.TOP
H = TOP + LH * len(LINES) + PAD
BG, TITLE = "#0d1117", "armguard-mcp · simulated FR3 (abridged)"
FONT_CANDIDATES = ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf", "DejaVuSansMono.ttf")
TYPING_STEPS = 3  # command lines (">>>") appear in a few chunks so the eye can follow
HOLD_LAST_S = 1.5
PACE = 0.85  # the SVG pauses sum to ~15 s; the typing frames add a little, so run slightly faster


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()  # type: ignore[return-value]


def _frame(visible: list[tuple[str, str]], font: ImageFont.FreeTypeFont, small: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for cx, color in ((20, "#ff5f57"), (38, "#febc2e"), (56, "#28c840")):
        d.ellipse((cx - 5, 11, cx + 5, 21), fill=color)
    d.text((W // 2, 16), TITLE, fill=DIM, font=small, anchor="mm")
    for i, (text, color) in enumerate(visible):
        d.text((PAD, TOP + LH * i - 13), text, fill=color, font=font)
    return img


def render() -> tuple[list[Image.Image], list[int]]:
    font, small = _font(13), _font(12)
    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, str]] = []
    for text, color, pause in LINES:
        # hold the previous frame for this line's pause, then reveal it
        if frames:
            durations[-1] += int(pause * PACE * 1000)
        else:
            frames.append(_frame([], font, small))
            durations.append(int(pause * PACE * 1000))
        if text.startswith(">>>") and len(text) > 8:
            cut = [len(text) * k // TYPING_STEPS for k in range(1, TYPING_STEPS)]
            for c in cut:
                frames.append(_frame([*shown, (text[:c] + "▏", color)], font, small))
                durations.append(70)
        shown.append((text, color))
        frames.append(_frame(list(shown), font, small))
        durations.append(0)
    durations[-1] += int(HOLD_LAST_S * 1000)
    return frames, durations


if __name__ == "__main__":
    frames, durations = render()
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    total = sum(durations) / 1000
    print(f"wrote {OUT.relative_to(ROOT)}: {len(frames)} frames, {total:.1f} s, {OUT.stat().st_size / 1024:.0f} KiB")
