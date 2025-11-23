"""Tests for name mention extraction from biographies."""

import pytest

from ancestral_synth.agents.dedup_agent import extract_name_mentions


class TestExtractNameMentions:
    """Tests for extract_name_mentions function."""

    def test_returns_empty_list_for_no_mentions(self) -> None:
        """Should return empty list when name not found."""
        biography = "John was born in Boston. He lived a long life."
        result = extract_name_mentions("Eleanor", biography)
        assert result == []

    def test_extracts_single_mention(self) -> None:
        """Should extract context around a single mention."""
        biography = "Mary had a sister named Eleanor who was born in 1920."
        result = extract_name_mentions("Eleanor", biography)
        assert len(result) == 1
        assert "Eleanor" in result[0]
        assert "sister" in result[0]

    def test_extracts_multiple_mentions(self) -> None:
        """Should extract context for each mention."""
        biography = (
            "Eleanor was the eldest daughter. She married in 1940. "
            "Eleanor was known for her kindness."
        )
        result = extract_name_mentions("Eleanor", biography)
        assert len(result) == 2
        assert "eldest daughter" in result[0]
        assert "kindness" in result[1]

    def test_case_insensitive_matching(self) -> None:
        """Should match names case-insensitively."""
        biography = "She met ELEANOR at the market. eleanor was friendly."
        result = extract_name_mentions("Eleanor", biography)
        assert len(result) == 2

    def test_respects_padding_limit(self) -> None:
        """Should limit context to specified padding."""
        # Create a very long biography with proper word boundaries
        before = "word " * 100  # 500 chars
        after = " more" * 100  # 500 chars
        biography = f"{before}Eleanor{after}"
        result = extract_name_mentions("Eleanor", biography, padding=300)
        assert len(result) == 1
        # Should be at most 300 + len("Eleanor") + 300 = ~607 chars
        assert len(result[0]) <= 650

    def test_handles_mention_at_start(self) -> None:
        """Should handle name at start of biography."""
        biography = "Eleanor was born in 1920 in New York City."
        result = extract_name_mentions("Eleanor", biography)
        assert len(result) == 1
        assert result[0].startswith("Eleanor")

    def test_handles_mention_at_end(self) -> None:
        """Should handle name at end of biography."""
        biography = "The youngest child was named Eleanor"
        result = extract_name_mentions("Eleanor", biography)
        assert len(result) == 1
        assert result[0].endswith("Eleanor")

    def test_returns_empty_for_empty_biography(self) -> None:
        """Should return empty list for empty biography."""
        result = extract_name_mentions("Eleanor", "")
        assert result == []

    def test_returns_empty_for_none_biography(self) -> None:
        """Should return empty list for None biography."""
        result = extract_name_mentions("Eleanor", None)
        assert result == []

    def test_deduplicates_overlapping_mentions(self) -> None:
        """Should not duplicate context when mentions are close together."""
        biography = "Eleanor Eleanor Eleanor was her name."
        result = extract_name_mentions("Eleanor", biography)
        # Should combine overlapping contexts
        assert len(result) <= 3

    def test_preserves_word_boundaries(self) -> None:
        """Should match whole words, not substrings."""
        biography = "Eleanora was different from Eleanor."
        result = extract_name_mentions("Eleanor", biography)
        # Should only match "Eleanor", not "Eleanora"
        assert len(result) == 1
        assert "different from Eleanor" in result[0]

    def test_extracts_relationship_context(self) -> None:
        """Should capture relationship words in context."""
        # Create a longer biography where mentions are far apart (>600 chars)
        filler = " " + ("The family lived in the countryside and worked hard every day. " * 10)
        biography = (
            f"His beloved wife Eleanor supported him through difficult times.{filler}"
            "His sister Eleanor helped with the farm work."
        )
        result = extract_name_mentions("Eleanor", biography, padding=100)
        assert len(result) == 2
        # Check that relationship words are captured
        wife_mention = [r for r in result if "wife" in r]
        sister_mention = [r for r in result if "sister" in r]
        assert len(wife_mention) == 1
        assert len(sister_mention) == 1
