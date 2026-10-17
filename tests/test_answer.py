"""Check prompt budgets without downloading or loading a language model."""

from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from transformers.tokenization_utils_base import (
    BatchEncoding, PreTrainedTokenizerBase)

from src.answer.answer import _build_promt
from src.dataset_classes import DataChunk, MinimalSource
from src.errors import SearchException


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    """Count each character as one token, including chat-template overhead."""
    mock = Mock(spec=PreTrainedTokenizerBase)

    def render(messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Add visible template overhead to the supplied user prompt."""
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(text: str, **kwargs: Any) -> BatchEncoding:
        """Return tensors with predictable token counts."""
        return BatchEncoding({"input_ids": torch.zeros((1, len(text)),
                                                       dtype=torch.long)})

    mock.apply_chat_template.side_effect = render
    mock.side_effect = encode
    return cast(PreTrainedTokenizerBase, mock)


def _chunk(index: int, content: str) -> DataChunk:
    """Make a source excerpt with a matching character span."""
    return DataChunk(index=index, content=content, min_source=MinimalSource(
        file_path="example.py", first_character_index=0,
        last_character_index=len(content)))


@pytest.mark.parametrize("reserved", [250, 256])
def test_last_source_cannot_overflow_prompt(
    tokenizer: PreTrainedTokenizerBase, reserved: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the fitting prefix when the final excerpt would exceed context."""
    chunks = [_chunk(0, "a" * 100), _chunk(1, "b" * 200)]
    complete = _build_promt("Question?", chunks, tokenizer, 10000, reserved)
    limit = int(complete["input_ids"].shape[1]) + reserved - 1
    fitting = _build_promt("Question?", chunks[:1], tokenizer, limit, reserved)
    actual = _build_promt("Question?", chunks, tokenizer, limit, reserved)
    assert actual["input_ids"].shape == fitting["input_ids"].shape
    assert actual["input_ids"].shape[1] + reserved <= limit
    assert "omitted" in capsys.readouterr().err


@pytest.mark.parametrize("reserved", [250, 256])
def test_exact_budget_keeps_source(
    tokenizer: PreTrainedTokenizerBase, reserved: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Accept a prompt that fits exactly after reserving generation tokens."""
    chunks = [_chunk(0, "a" * 100)]
    complete = _build_promt("Question?", chunks, tokenizer, 10000, reserved)
    limit = int(complete["input_ids"].shape[1]) + reserved
    actual = _build_promt("Question?", chunks, tokenizer, limit, reserved)
    assert actual["input_ids"].shape[1] + reserved == limit
    assert capsys.readouterr().err == ""


def test_overlong_question_is_rejected(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Reject an oversized question even when no sources are supplied."""
    with pytest.raises(SearchException, match="Question is too long"):
        _build_promt("word " * 1000, [], tokenizer, 2000, 256)


def test_no_source_fits_is_rejected(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Avoid silently generating without any of the retrieved evidence."""
    empty = _build_promt("Question?", [], tokenizer, 10000, 256)
    limit = int(empty["input_ids"].shape[1]) + 256 + 1
    with pytest.raises(SearchException, match="No retrieved source fits"):
        _build_promt("Question?", [_chunk(0, "a" * 100)],
                     tokenizer, limit, 256)
