"""Check configurable recall cutoffs, averaging, and CLI output."""

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src.dataset_classes import (
    AnsweredQuestion, MinimalSearchResults, MinimalSource,
    RagDataset, StudentSearchResults)
from src.errors import SearchException
from src.evaluate.evaluate import _recall_at_k, evaluate


@pytest.fixture
def evaluation_files(tmp_path: Path) -> tuple[Path, Path]:
    """Put the only matching source at rank six in ten saved results."""
    correct = MinimalSource(file_path="correct.py", first_character_index=0,
                            last_character_index=100)
    wrong = correct.model_copy(update={"file_path": "wrong.py"})
    results = StudentSearchResults(k=10, search_results=[MinimalSearchResults(
        question_id="q1", question="Where?",
        retrieved_sources=[wrong] * 5 + [correct] + [wrong] * 4)])
    reference = RagDataset(rag_questions=[AnsweredQuestion(
        question_id="q1", question="Where?",
        sources=[correct], answer="Here")])
    result_path = tmp_path / "results.json"
    reference_path = tmp_path / "ref.json"
    result_path.write_text(results.model_dump_json(), encoding="utf-8")
    reference_path.write_text(reference.model_dump_json(), encoding="utf-8")
    return result_path, reference_path


@pytest.mark.parametrize("k, expected", [
    (None, "Recall_10 = 1.0"), (5, "Recall_5 = 0.0"), (6, "Recall_6 = 1.0"),
])
def test_evaluation_uses_requested_or_saved_k(
    evaluation_files: tuple[Path, Path], k: int | None, expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Respect the saved retrieval count and an explicit smaller cutoff."""
    results, reference = evaluation_files
    evaluate(str(results), str(reference), k=k)
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.parametrize("k", [0, -1, True, False, 1.5, "5", 11])
def test_evaluation_rejects_invalid_cutoff(
    evaluation_files: tuple[Path, Path], k: Any,
) -> None:
    """Reject invalid counts and cutoffs beyond the saved ranking depth."""
    results, reference = evaluation_files
    with pytest.raises(SearchException, match="--k"):
        evaluate(str(results), str(reference), k=k)


def test_recall_averages_per_question() -> None:
    """Give questions equal weight despite different source counts."""
    first = MinimalSource(file_path="first.py", first_character_index=0,
                          last_character_index=100)
    second = first.model_copy(update={"file_path": "second.py"})
    reference = RagDataset(rag_questions=[
        AnsweredQuestion(question_id="q1", question="First?",
                         sources=[first, second], answer="First"),
        AnsweredQuestion(question_id="q2", question="Second?",
                         sources=[second], answer="Second"),
    ])
    results = StudentSearchResults(k=1, search_results=[
        MinimalSearchResults(question_id="q1", question="First?",
                             retrieved_sources=[first]),
        MinimalSearchResults(question_id="q2", question="Second?",
                             retrieved_sources=[second]),
    ])
    assert _recall_at_k(results, reference, 1) == pytest.approx(0.75)


@pytest.mark.parametrize("start, path, expected", [
    (95, "correct.py", 1.0), (96, "correct.py", 0.0), (95, "wrong.py", 0.0),
])
def test_recall_keeps_exact_paths_and_iou_threshold(
    start: int, path: str, expected: float,
) -> None:
    """Require the same path and accept overlap exactly at IoU 0.05."""
    source = MinimalSource(file_path="correct.py", first_character_index=0,
                           last_character_index=100)
    retrieved = source.model_copy(update={
        "file_path": path, "first_character_index": start})
    reference = RagDataset(rag_questions=[AnsweredQuestion(
        question_id="q1", question="Where?", sources=[source], answer="Here")])
    results = StudentSearchResults(k=1, search_results=[MinimalSearchResults(
        question_id="q1", question="Where?", retrieved_sources=[retrieved])])
    assert _recall_at_k(results, reference, 1) == expected


def test_evaluate_cli_accepts_cutoff(
    evaluation_files: tuple[Path, Path],
) -> None:
    """Expose the cutoff through Fire without changing the required paths."""
    results, reference = evaluation_files
    completed = subprocess.run([
        sys.executable, "-m", "src", "evaluate",
        "--student_search_results_path", str(results),
        "--dataset_path", str(reference), "--k", "5",
    ], capture_output=True, text=True, timeout=30,
        cwd=Path(__file__).resolve().parent.parent)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Recall_5 = 0.0"
