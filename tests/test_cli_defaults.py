"""The bundled policy and the CLI's policy defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from armguard_mcp.cli import build_parser, main, resolve_policy_path
from armguard_mcp.policy import bundled_policy_path, load_policy

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_policy_is_a_copy_of_the_example() -> None:
    bundled = bundled_policy_path("fr3")
    assert bundled.read_text(encoding="utf-8") == (ROOT / "examples" / "policies" / "fr3.yaml").read_text(encoding="utf-8")
    assert load_policy(bundled).robot == load_policy(ROOT / "examples" / "policies" / "fr3.yaml").robot


def test_fake_backend_defaults_to_the_bundled_policy() -> None:
    args = build_parser().parse_args([])
    assert args.policy is None and args.backend == "fake"
    assert resolve_policy_path(args.policy, args.backend) == str(bundled_policy_path("fr3"))
    assert resolve_policy_path("my.yaml", "fake") == "my.yaml"


def test_ros2_backend_requires_an_explicit_policy(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="--policy is required with --backend ros2"):
        resolve_policy_path(None, "ros2")
    assert main(["--backend", "ros2"]) == 2
    assert "--policy is required" in capsys.readouterr().err
