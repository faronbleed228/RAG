"""Check dataset serialization, question alignment, and batch failures."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import bm25s
from pydantic import ValidationError
import pytest

from src.errors import SearchException
from src.search.search import _load_chunks, search
from src.search.search_dataset import StudentSearchResults, search_dataset


@pytest.mark.parametrize("k", [1, 3])
def test_dataset_output_preserves_questions_and_sources(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path, k: int
) -> None:
    """Write the required envelope and keep rankings tied to their queries."""
    output = tmp_path / "nested" / "results with spaces"
    search_dataset(
        dataset_path=str(dataset_path), k=k,
        save_directory=str(output), index_directory=str(indexed_dir),
    )
    saved_path = output / dataset_path.name
    raw = json.loads(saved_path.read_text(encoding="utf-8"))
    assert set(raw) == {"search_results", "k"}
    saved = StudentSearchResults.model_validate(raw)
    questions = json.loads(dataset_path.read_text())["rag_questions"]
    assert saved.k == k
    assert len(saved.search_results) == len(questions)

    chunks = _load_chunks(indexed_dir / "chunks.jsonl", k=1)
    locations = {
        (c.min_source.file_path,
         c.min_source.first_character_index,
         c.min_source.last_character_index): c
        for c in chunks
    }
    project_root = Path(__file__).resolve().parent.parent
    expected_files = ["dogs.md", "cats.md", "birds.py"]
    for question, result, filename in zip(
        questions, saved.search_results, expected_files, strict=True
    ):
        assert result.question_id == question["question_id"]
        assert result.question == question["question"]
        assert len(result.retrieved_sources) == k
        assert result.retrieved_sources[0].file_path.endswith("/" + filename)
        for source in result.retrieved_sources:
            start = source.first_character_index
            end = source.last_character_index
            chunk = locations[(source.file_path, start, end)]
            content = (project_root / source.file_path).read_text(
                encoding="utf-8"
            )
            assert 0 <= start < end <= len(content)
            assert end - start <= 2000
            assert content[start:end] == chunk.content


def test_dataset_matches_individual_searches(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path
) -> None:
    """Produce the same ranked sources as individual searches."""
    output = tmp_path / "results"
    search_dataset(
        str(dataset_path), k=1, save_directory=str(output),
        index_directory=str(indexed_dir),
    )
    saved = StudentSearchResults.model_validate_json(
        (output / dataset_path.name).read_text(encoding="utf-8")
    )
    for result in saved.search_results:
        chunks, documents = search(
            result.question, k=1, index_dir=str(indexed_dir),
        )
        expected = [chunks[int(index)].min_source for index in documents[0]]
        assert result.retrieved_sources == expected


def test_dataset_loads_index_and_chunks_once(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path
) -> None:
    """Avoid reloading persisted data for each question in a batch."""
    with patch(
        "src.search.search.bm25s.BM25.load", wraps=bm25s.BM25.load
    ) as index_load, patch(
        "src.search.search._load_chunks", wraps=_load_chunks
    ) as chunks_load:
        search_dataset(
            str(dataset_path), k=1, save_directory=str(tmp_path / "results"),
            index_directory=str(indexed_dir),
        )
    assert index_load.call_count == 1
    assert chunks_load.call_count == 1


def test_dataset_accepts_answered_questions(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path
) -> None:
    """Search answered questions without copying their reference sources."""
    raw = json.loads(dataset_path.read_text())
    for question in raw["rag_questions"]:
        question.update(answer="Reference answer", sources=[])
    dataset_path.write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "results"
    search_dataset(
        str(dataset_path), k=1, save_directory=str(output),
        index_directory=str(indexed_dir),
    )
    saved = StudentSearchResults.model_validate_json(
        (output / dataset_path.name).read_text(encoding="utf-8")
    )
    assert [q.question_id for q in saved.search_results] == [
        q["question_id"] for q in raw["rag_questions"]
    ]
    assert all(len(q.retrieved_sources) == 1 for q in saved.search_results)


def test_dataset_generates_missing_question_id(
    tmp_path: Path, indexed_dir: Path
) -> None:
    """Assign a UUID when the input question has no explicit ID."""
    path = tmp_path / "no-id.json"
    path.write_text(json.dumps({"rag_questions": [{"question": "purring"}]}))
    output = tmp_path / "results"
    search_dataset(
        str(path), k=1, save_directory=str(output),
        index_directory=str(indexed_dir),
    )
    saved = StudentSearchResults.model_validate_json(
        (output / path.name).read_text(encoding="utf-8")
    )
    assert len(saved.search_results) == 1
    question_id = saved.search_results[0].question_id
    assert str(UUID(question_id)) == question_id


@pytest.mark.parametrize("value", ["", " \t\n", 123, True, [], {}])
@pytest.mark.parametrize("argument", ["dataset_path", "save_directory"])
def test_invalid_dataset_paths(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path,
    argument: str, value: Any,
) -> None:
    """Distinguish explicit bad paths from omitted optional arguments."""
    arguments: dict[str, Any] = {
        "dataset_path": str(dataset_path), "k": 1,
        "save_directory": str(tmp_path / "results"),
        "index_directory": str(indexed_dir),
    }
    arguments[argument] = value
    with pytest.raises(SearchException):
        search_dataset(**arguments)


@pytest.mark.parametrize("k", [0, -1, 101, True, 1.0, 1.5, "abc", None])
def test_invalid_dataset_k(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path, k: Any
) -> None:
    """Validate the batch count before starting retrieval."""
    with pytest.raises(SearchException, match="--k"):
        search_dataset(
            str(dataset_path), k=k, save_directory=str(tmp_path / "results"),
            index_directory=str(indexed_dir),
        )


def test_dataset_k_above_index_size(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path
) -> None:
    """Reject counts above the actual index size, not only above 100."""
    count = len(_load_chunks(indexed_dir / "chunks.jsonl", k=1))
    with pytest.raises(SearchException, match="--k"):
        search_dataset(
            str(dataset_path), k=count + 1,
            save_directory=str(tmp_path / "results"),
            index_directory=str(indexed_dir),
        )


def test_missing_dataset(tmp_path: Path) -> None:
    """Explain an absent dataset before trying to load an index."""
    with pytest.raises(SearchException, match="dataset"):
        search_dataset(
            str(tmp_path / "missing.json"), k=1,
            save_directory=str(tmp_path / "results"),
        )


@pytest.mark.parametrize("text", [
    "{broken json", "{}", '{"rag_questions": "wrong type"}',
    '{"rag_questions": [{"question_id": "missing-question"}]}',
])
def test_malformed_dataset(tmp_path: Path, text: str) -> None:
    """Reject invalid JSON and invalid schemas without writing results."""
    path = tmp_path / "broken.json"
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "results"
    with pytest.raises(ValidationError):
        search_dataset(str(path), k=1, save_directory=str(output))
    assert not (output / path.name).exists()


@pytest.mark.parametrize("question", ["", " \t\n"])
def test_blank_dataset_question(
    tmp_path: Path, indexed_dir: Path, question: str
) -> None:
    """Reject a blank member of an otherwise valid question batch."""
    path = tmp_path / "blank-question.json"
    path.write_text(json.dumps({"rag_questions": [
        {"question": "purring"}, {"question": question},
    ]}), encoding="utf-8")
    with pytest.raises(SearchException, match="query"):
        search_dataset(
            str(path), k=1, save_directory=str(tmp_path / "results"),
            index_directory=str(indexed_dir),
        )


def test_empty_dataset_is_rejected(
    tmp_path: Path, indexed_dir: Path
) -> None:
    """Reject an empty question batch without producing a result file."""
    path = tmp_path / "empty.json"
    path.write_text('{"rag_questions": []}', encoding="utf-8")
    output = tmp_path / "results"
    with pytest.raises(SearchException, match="Empty"):
        search_dataset(
            str(path), k=1, save_directory=str(output),
            index_directory=str(indexed_dir),
        )
    assert not (output / path.name).exists()
