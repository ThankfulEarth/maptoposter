"""Tests for resolve_title_fonts: title-font resolution with Roboto fallback."""

from __future__ import annotations

from pathlib import Path

from create_map_poster import resolve_title_fonts


def test_falls_back_to_roboto_when_none(tmp_path: Path) -> None:
    fonts = resolve_title_fonts(None, fonts_root=str(tmp_path))
    assert fonts["bold"].endswith("Roboto-Bold.ttf")
    assert fonts["regular"].endswith("Roboto-Regular.ttf")
    assert fonts["light"].endswith("Roboto-Light.ttf")


def test_uses_generated_when_present(tmp_path: Path) -> None:
    gen = tmp_path / "generated" / "oswald"
    gen.mkdir(parents=True)
    for role in ("bold", "regular", "light"):
        (gen / f"{role}.ttf").write_bytes(b"x")
    fonts = resolve_title_fonts("oswald", fonts_root=str(tmp_path))
    assert fonts["bold"] == str(gen / "bold.ttf")
