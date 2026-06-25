"""Tests for constants module."""

from __future__ import annotations

from constants import (
    SEGMENT_COLORS,
    SEGMENT_ORDER,
    SEGMENT_ICONS,
)


def test_segment_order_has_five_segments() -> None:
    assert len(SEGMENT_ORDER) == 5


def test_segment_colors_cover_all_segments() -> None:
    for seg in SEGMENT_ORDER:
        assert seg in SEGMENT_COLORS, f"Missing colour for {seg}"


def test_segment_icons_cover_all_segments() -> None:
    for seg in SEGMENT_ORDER:
        assert seg in SEGMENT_ICONS, f"Missing icon for {seg}"


def test_segment_order_matches_color_keys() -> None:
    assert SEGMENT_ORDER == list(SEGMENT_COLORS.keys())
