"""Render docs/demo.svg: an animated, abridged terminal view of `python scripts/demo_fake.py`.

Every line below is shortened from the real demo output (full transcript: docs/demo.md).
Re-run after changing the demo:  python scripts/make_demo_svg.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.svg"

FG, DIM, CMD, OK, WARN, BAD = "#d4d4d4", "#8a8a8a", "#6cb6ff", "#7ee787", "#e3b341", "#ff7b72"

# (text, color, pause before this line in seconds)
LINES: list[tuple[str, str, float]] = [
    ("$ python scripts/demo_fake.py        # an LLM client talking to armguard-mcp", DIM, 0.4),
    ("", FG, 0.2),
    (">>> plan_to_pose(x=0.40, y=0.10, z=0.40)", CMD, 0.8),
    ('    status: "executable"   violations: []', OK, 0.6),
    (">>> execute_plan(ecfd2392c028)", CMD, 0.7),
    ('    status: "completed"    approval: "policy"   (small move, inside the envelope)', OK, 0.8),
    ("", FG, 0.2),
    (">>> plan_to_joints(joint 1 turns -0.9 rad)", CMD, 0.8),
    ('    status: "needs_approval"   LARGE_MOTION (soft): 0.900 rad > 0.8 rad', WARN, 0.8),
    (">>> execute_plan(3a6f9dd9e508)", CMD, 0.6),
    ("    ┌ APPROVE ROBOT MOTION on 'fr3'?  3.61 s, peak joint speed 10% of limit", WARN, 0.5),
    ("    │ Force monitoring: on - aborts above 25.0 N / 5.0 N*m", WARN, 0.3),
    ("    └ human approves (operator=lynn)", WARN, 0.9),
    ('    status: "completed"    approval: "human"', OK, 0.8),
    ("", FG, 0.2),
    (">>> plan_to_pose(x=0.62, y=0.42, z=0.30)     # into the 'camera_mount' keep-out zone", CMD, 0.9),
    ("    status: \"rejected\"   KEEP_OUT (hard): TCP enters keep-out zone 'camera_mount'", BAD, 0.8),
    (">>> execute_plan(36b751dac73c)               # the model tries anyway", CMD, 0.8),
    ("    refused: plan violates hard safety limits and can never be executed", BAD, 0.9),
    ("", FG, 0.2),
    ('>>> estop(reason="operator saw something odd")', CMD, 0.8),
    ("    estopped: true", BAD, 0.6),
    (">>> plan_to_pose(x=0.40, y=0.00, z=0.40)", CMD, 0.7),
    ("    refused: software e-stop is active; call reset_estop (requires human approval)", BAD, 0.8),
]

W, PAD, LH, TOP = 880, 18, 20, 46


def render() -> str:
    h = TOP + LH * len(LINES) + PAD
    t, rows, styles = 0.0, [], []
    for i, (text, color, pause) in enumerate(LINES):
        t += pause
        styles.append(f".l{i}{{animation-delay:{t:.2f}s}}")
        y = TOP + LH * i
        rows.append(
            f'<text class="l l{i}" x="{PAD}" y="{y}" fill="{color}" xml:space="preserve">{escape(text)}</text>'
        )
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        'role="img" aria-label="armguard-mcp demo: an in-envelope move runs, a large move needs human '
        'approval, a keep-out request is rejected and cannot be executed, and the e-stop blocks planning">'
    )
    css = (
        "<style>"
        ".l{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;opacity:0;"
        "animation:in .25s ease-out forwards}"
        "@keyframes in{to{opacity:1}}"
        "@media (prefers-reduced-motion:reduce){.l{animation:none;opacity:1}}" + "".join(styles) + "</style>"
    )
    chrome = (
        f'<rect width="{W}" height="{h}" rx="8" fill="#0d1117"/>'
        '<circle cx="20" cy="16" r="5" fill="#ff5f57"/><circle cx="38" cy="16" r="5" fill="#febc2e"/>'
        '<circle cx="56" cy="16" r="5" fill="#28c840"/>'
        f'<text x="{W // 2}" y="20" fill="{DIM}" text-anchor="middle" '
        'style="font:12px ui-monospace,Menlo,monospace">armguard-mcp · simulated FR3 (abridged)</text>'
    )
    return head + css + chrome + "".join(rows) + "</svg>\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
