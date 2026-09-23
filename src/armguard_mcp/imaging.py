"""Tiny image helpers: a pure-Python PNG encoder and optional Pillow-based downscaling."""

from __future__ import annotations

import io
import struct
import zlib
from collections.abc import Sequence


def encode_png_rgb(width: int, height: int, rgb_rows: Sequence[bytes]) -> bytes:
    """Encode 8-bit RGB rows (``3 * width`` bytes each) as a PNG file."""
    if len(rgb_rows) != height or any(len(r) != 3 * width for r in rgb_rows):
        raise ValueError("row data does not match width/height")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row for row in rgb_rows)  # filter type 0 (None) per scanline
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 = RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def gradient_png(width: int = 64, height: int = 48, seed: int = 0) -> bytes:
    """Deterministic test pattern: a colour gradient with a dark square 'object' in the middle."""
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            r = (x * 255) // max(1, width - 1)
            g = (y * 255) // max(1, height - 1)
            b = (seed * 37) % 256
            if width // 3 <= x < 2 * width // 3 and height // 3 <= y < 2 * height // 3:
                r, g, b = 30, 30, 30
            row += bytes((r, g, b))
        rows.append(bytes(row))
    return encode_png_rgb(width, height, rows)


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def downscale(data: bytes, max_width: int) -> tuple[bytes, str, int, int]:
    """Downscale an encoded image to ``max_width`` with Pillow; returns (bytes, mime, w, h)."""
    from PIL import Image  # optional dependency: pip install armguard-mcp[image]

    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        new_h = max(1, round(h * max_width / w))
        small = im.convert("RGB").resize((max_width, new_h))
        out = io.BytesIO()
        small.save(out, format="PNG")
        return out.getvalue(), "image/png", max_width, new_h
