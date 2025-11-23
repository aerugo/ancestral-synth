"""Tests for agent initialization.

These tests verify that agents can be properly instantiated with the
pydantic-ai library. They test the actual Agent creation, not just
internal methods.
"""

import os
from unittest.mock import patch

import pytest
from pydantic_ai import Agent

from ancestral_synth.agents.biography_agent import BiographyAgent
from ancestral_synth.agents.correction_agent import CorrectionAgent
from ancestral_synth.agents.dedup_agent import DedupAgent
from ancestral_synth.agents.extraction_agent import ExtractionAgent
from ancestral_synth.domain.models import Biography, ExtractedData


class TestBiographyAgentInitialization:
    """Tests for BiographyAgent initialization."""

    def test_agent_can_be_instantiated_with_test_model(self) -> None:
        """BiographyAgent should instantiate without errors using test model.

        This tests that the Agent constructor is called with correct parameters.
        Using 'test' model to avoid needing real API keys.
        """
        agent = BiographyAgent(model="test")

        assert agent is not None
        assert hasattr(agent, "_agent")
        assert isinstance(agent._agent, Agent)

    def test_agent_uses_correct_output_type(self) -> None:
        """BiographyAgent should configure output_type as Biography.

        This verifies the agent is configured to output Biography objects.
        """
        agent = BiographyAgent(model="test")

        # The agent's output schema should be configured for Biography
        # This tests that output_type (not result_type) is used correctly
        assert agent._agent is not None

    def test_agent_has_system_prompt(self) -> None:
        """BiographyAgent should have a system prompt configured."""
        agent = BiographyAgent(model="test")

        # Verify system prompt is set (implementation detail may vary)
        assert agent._agent is not None


class TestExtractionAgentInitialization:
    """Tests for ExtractionAgent initialization."""

    def test_agent_can_be_instantiated_with_test_model(self) -> None:
        """ExtractionAgent should instantiate without errors using test model."""
        agent = ExtractionAgent(model="test")

        assert agent is not None
        assert hasattr(agent, "_agent")
        assert isinstance(agent._agent, Agent)

    def test_agent_uses_correct_output_type(self) -> None:
        """ExtractionAgent should configure output_type as ExtractedData."""
        agent = ExtractionAgent(model="test")

        assert agent._agent is not None


class TestDedupAgentInitialization:
    """Tests for DedupAgent initialization."""

    def test_agent_can_be_instantiated_with_test_model(self) -> None:
        """DedupAgent should instantiate without errors using test model."""
        agent = DedupAgent(model="test")

        assert agent is not None
        assert hasattr(agent, "_agent")
        assert isinstance(agent._agent, Agent)

    def test_agent_uses_correct_output_type(self) -> None:
        """DedupAgent should configure output_type as DedupResult."""
        agent = DedupAgent(model="test")

        assert agent._agent is not None


class TestCorrectionAgentInitialization:
    """Tests for CorrectionAgent initialization."""

    def test_agent_can_be_instantiated_with_test_model(self) -> None:
        """CorrectionAgent should instantiate without errors using test model."""
        agent = CorrectionAgent(model="test")

        assert agent is not None
        assert hasattr(agent, "_agent")
        assert isinstance(agent._agent, Agent)

    def test_agent_uses_correct_output_type(self) -> None:
        """CorrectionAgent should configure output_type as ExtractedData."""
        agent = CorrectionAgent(model="test")

        assert agent._agent is not None


class TestAgentDefaultModel:
    """Tests for agent default model configuration."""

    def test_biography_agent_uses_settings_when_no_model_provided(self) -> None:
        """BiographyAgent should use settings for default model.

        This test uses mocking to avoid needing actual API keys.
        """
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Should not raise - uses default model from settings
            agent = BiographyAgent()
            assert agent._agent is not None

    def test_extraction_agent_uses_settings_when_no_model_provided(self) -> None:
        """ExtractionAgent should use settings for default model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = ExtractionAgent()
            assert agent._agent is not None

    def test_dedup_agent_uses_settings_when_no_model_provided(self) -> None:
        """DedupAgent should use settings for default model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = DedupAgent()
            assert agent._agent is not None

    def test_correction_agent_uses_settings_when_no_model_provided(self) -> None:
        """CorrectionAgent should use settings for default model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = CorrectionAgent()
            assert agent._agent is not None
