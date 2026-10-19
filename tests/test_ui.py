"""Tests for the pure HTML/CSS helpers in ui.py."""

from __future__ import annotations

import ui
from analytics import fmt_money, priority_label


def test_flat_escapes_dollar_so_currency_is_not_latex() -> None:
    assert "$" not in ui._flat("<div>$1.2k and $3k</div>")
    assert "&#36;1.2k" in ui._flat("<div>$1.2k</div>")


def test_kpi_card_text_variant_and_escaping() -> None:
    card = ui.kpi_card("Segment", "<b>x</b>", "sub", text=True)
    assert "kpi-value is-text" in card
    assert "&lt;b&gt;x&lt;/b&gt;" in card  # value is escaped, sub is trusted HTML


def test_grid_rules_cover_every_card_count_and_size() -> None:
    css = ui._grid_rules()
    for n in (2, 3, 4, 5, 6):
        assert f'data-n="{n}"][data-size="sm"]' in css
    for n in (2, 3, 4):
        assert f'data-n="{n}"][data-size="lg"]' in css
    # the 1:2 layout must out-rank the generic 2-up rule (specificity regression)
    assert 'data-n="2"][data-size="lg"][data-layout="1-2"]' in css


def test_grid_ladders_are_ascending_and_never_orphan_a_single_card() -> None:
    for by_n in ui._LADDERS.values():
        for n, ladder in by_n.items():
            widths = [w for w, _ in ladder]
            assert widths == sorted(widths)
            for _, cols in ladder:
                if n != 5:  # 5 is prime: any 2-4 column split leaves a short last row
                    assert n % cols != 1 or n == 1, f"{n} cards in {cols} columns leaves one orphan"


def test_priority_label_keeps_acronyms() -> None:
    assert priority_label("ASSESS ROI") == "Assess ROI"
    assert priority_label("WIN BACK") == "Win back"
    assert priority_label("PROTECT") == "Protect"
    assert priority_label("") == ""


def test_fmt_money_rounding_boundary() -> None:
    assert fmt_money(999_600) == "£1.00M"  # not "£1000k"
    assert fmt_money(842) == "£842"
    assert fmt_money(4_300) == "£4.3k"
    assert fmt_money(-26_000) == "-£26k"
