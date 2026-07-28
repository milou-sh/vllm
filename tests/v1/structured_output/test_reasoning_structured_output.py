# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


class MockReasoner:
    def __init__(self, tokenizer):
        self.is_reasoning_end = Mock(return_value=False)
        self.is_reasoning_end_streaming = Mock(return_value=False)


class TestReasoningStructuredOutput:
    """Test reasoning-aware structured output functionality."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig."""
        config = Mock(spec=ModelConfig)
        config.skip_tokenizer_init = True  # Skip tokenizer init to avoid network calls
        config.get_vocab_size = Mock(return_value=50000)
        # Add missing runner_type attribute that tokenizer initialization expects
        config.runner_type = "generate"
        # Add other attributes that tokenizer initialization might need
        config.tokenizer = "test-tokenizer"
        config.tokenizer_mode = "auto"
        config.trust_remote_code = False
        config.tokenizer_revision = None
        return config

    @pytest.fixture
    def mock_scheduler_config(self):
        """Create a mock SchedulerConfig."""
        config = Mock(spec=SchedulerConfig)
        config.max_num_seqs = 128
        return config

    @pytest.fixture
    def mock_vllm_config(self, mock_model_config, mock_scheduler_config):
        """Create a mock VllmConfig."""
        config = Mock(spec=VllmConfig)
        config.model_config = mock_model_config
        config.scheduler_config = mock_scheduler_config
        config.structured_outputs_config = Mock()
        config.structured_outputs_config.reasoning_parser = None
        config.structured_outputs_config.enable_in_reasoning = False
        config.speculative_config = None
        return config

    @pytest.fixture
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.reasoning_parser_kwargs = None
        request.structured_output_request.reasoner = None
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
        return request

    @pytest.fixture
    def manager_with_reasoner(self, mock_vllm_config):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = MockReasoner
        manager.tokenizer = Mock()
        return manager

    def test_should_fill_bitmask_with_enable_in_reasoning(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_fill_bitmask(mock_request_with_structured_output)
        assert result is True

    def test_should_fill_bitmask_without_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = manager_with_reasoner.vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        result = manager_with_reasoner.should_fill_bitmask(
            mock_request_with_structured_output
        )

        # Should set reasoning_ended and return its value
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is False
        )
        assert result is False

    def test_should_fill_bitmask_no_reasoner(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when no reasoner is configured."""
        manager = StructuredOutputManager(mock_vllm_config)

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_fill_bitmask_uses_request_reasoning_parser_kwargs(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test request-level parser kwargs override the default reasoner."""

        class KwargReasoner:
            def __init__(self, tokenizer, chat_template_kwargs=None):
                self.chat_template_kwargs = chat_template_kwargs or {}

            def is_reasoning_end(self, input_ids):
                return not self.chat_template_kwargs.get("enable_thinking", False)

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = KwargReasoner
        manager.tokenizer = Mock()

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_parser_kwargs = {
            "chat_template_kwargs": {"enable_thinking": True}
        }

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        assert result is False
        assert (
            mock_request_with_structured_output.structured_output_request.reasoner
            is not None
        )

    def test_should_advance_with_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        manager_with_reasoner.enable_in_reasoning = True

        # Should always return True when enable_in_reasoning is enabled
        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has not ended."""
        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning ends in current step."""
        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoner = reasoner

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_just_ended_with_spec_decode_structural_tag(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When reasoning ends this step, advance immediately for structural
        tags with speculative decoding."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.STRUCTURAL_TAG,
            "{}",
        )
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock()

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_should_advance_reasoning_already_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has already ended."""
        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return True since reasoning has ended
        assert result is True

    def test_grammar_bitmask_does_not_advance_post_reasoning_drafts(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Reasoning ending inside a speculative window must not advance the
        FSM over the drafts that follow the marker.

        Those drafts were sampled before the grammar became active, so feeding
        them to ``accept_tokens`` logs a spurious "Failed to advance FSM" error
        on every reasoning boundary. The grammar is still constrained via the
        bitmask; only the (unreliable) advance is skipped.
        """
        import torch

        manager = manager_with_reasoner
        # Enable speculative decoding with a 3-token window.
        manager.vllm_config.speculative_config = Mock()
        manager.vllm_config.speculative_config.num_speculative_tokens = 3

        # Real bitmask tensor so _fill_bitmasks can write into it.
        manager.backend = Mock()
        manager.backend.allocate_token_bitmask = Mock(
            return_value=torch.zeros((512, 1563), dtype=torch.int32)
        )

        request = mock_request_with_structured_output
        request.structured_output_request.reasoning_ended = None
        grammar = request.structured_output_request.grammar
        grammar.accept_tokens = Mock(return_value=True)

        # Reasoning ends on the marker token (id 11), the middle of the window.
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end.return_value = False
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda _history, delta: list(delta) == [11]
        )
        request.structured_output_request.reasoner = reasoner

        req_id = "req-0"
        manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: [10, 11, 12]},
        )

        # The FSM must never be advanced over the un-constrained drafts.
        grammar.accept_tokens.assert_not_called()
