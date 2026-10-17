"""Validate answer datasets before model loading or generation."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from src.answer.answer_dataset import _load_search_results
from src.errors import SearchException


def _results_file(tmp_path: Path, k: Any = 1,
                  question: str = "Question?") -> Path:
    """Create a retrieval envelope with one source and configurable input."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"k": k, "search_results": [{
        "question_id": "q1", "question": question, "retrieved_sources": [{
            "file_path": "source.txt", "first_character_index": 0,
            "last_character_index": 10,
        }],
    }]}), encoding="utf-8")
    return path


@pytest.mark.parametrize("k", [0, -1, 101, True, False, 1.5, "1"])
def test_invalid_saved_k_is_rejected(tmp_path: Path, k: Any) -> None:
    """Reject invalid counts without coercing booleans, strings or floats."""
    with pytest.raises(ValidationError):
        _load_search_results(_results_file(tmp_path, k=k))


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_blank_saved_question_is_rejected(
    tmp_path: Path, question: str,
) -> None:
    """Reject blank questions before the answer pipeline can load Qwen3."""
    with pytest.raises(ValidationError):
        _load_search_results(_results_file(tmp_path, question=question))


def test_missing_sources_are_rejected(tmp_path: Path) -> None:
    """Require evidence for every question supplied to batch generation."""
    path = _results_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["search_results"][0]["retrieved_sources"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SearchException, match="retrieved source"):
        _load_search_results(path)


def test_valid_question_text_is_preserved(tmp_path: Path) -> None:
    """Keep original whitespace and question IDs for nonblank questions."""
    results = _load_search_results(_results_file(
        tmp_path, question="  Question?  "))
    assert results.search_results[0].question == "  Question?  "
    assert results.search_results[0].question_id == "q1"
