"""Tests for resolve_label_fonts: label-font resolution with Noto Sans default and Roboto fallback."""

from __future__ import annotations

from create_map_poster import resolve_label_fonts


def test_defaults_to_noto_sans_when_none():
    fonts = resolve_label_fonts(None)
    assert fonts["regular"].endswith("NotoSans-Regular.ttf")
    assert fonts["bold"].endswith("NotoSans-Bold.ttf")
    assert fonts["italic"].endswith("NotoSans-Italic.ttf")


def test_selects_noto_serif():
    fonts = resolve_label_fonts("noto_serif")
    assert fonts["regular"].endswith("NotoSerif-Regular.ttf")


def test_unknown_id_falls_back_to_noto_sans():
    fonts = resolve_label_fonts("does_not_exist")
    assert fonts["regular"].endswith("NotoSans-Regular.ttf")
