import pytest
from src.data.processor import DataProcessor


def test_normalize_text_removes_extra_spaces():
    result = DataProcessor.normalize_text("  HOLA   MUNDO!!  ")
    assert result == "HOLA MUNDO!!"


def test_normalize_text_handles_unicode():
    result = DataProcessor.normalize_text("café  naïve")
    assert "  " not in result


def test_normalize_text_empty():
    assert DataProcessor.normalize_text("") == ""


def test_normalize_text_whitespace_only():
    assert DataProcessor.normalize_text("   ") == ""


def test_normalize_text_preserves_punctuation():
    result = DataProcessor.normalize_text("¿Cómo estás? ¡Bien!")
    assert "¿" in result
    assert "?" in result
    assert "¡" in result
    assert "!" in result


def test_normalize_text_newlines_to_spaces():
    result = DataProcessor.normalize_text("line1\nline2\n\nline3")
    assert "\n" not in result
    assert result == "line1 line2 line3"


def test_normalize_text_tabs_to_spaces():
    result = DataProcessor.normalize_text("a\tb\t\tc")
    assert "\t" not in result
    assert result == "a b c"
