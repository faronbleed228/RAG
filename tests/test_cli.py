"""Check the actual Fire CLI in separate Python processes."""

from pathlib import Path
import subprocess
import sys

import pytest

from src.search.search_dataset import StudentSearchResults


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the module using the same environment as pytest."""
    return subprocess.run(
        [sys.executable, "-m", "src", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_help_does_not_start_indexing() -> None:
    """Show available commands without running corpus ingestion."""
    result = run_cli("--help")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "index" in output
    assert "search" in output
    assert "Chunking" not in output


def test_index_then_search(corpus_dir: Path, tmp_path: Path) -> None:
    """Save externally, then reload with a new process and find a source."""
    directory = tmp_path / "index with spaces"
    indexed = run_cli(
        "index", "--input_directory", str(corpus_dir),
        "--index_directory", str(directory),
    )
    assert indexed.returncode == 0, indexed.stderr
    assert (directory / "chunks.jsonl").is_file()
    assert (directory / "bm25" / "params.index.json").is_file()

    searched = run_cli(
        "search", "purring", "--k", "1", "--index_dir", str(directory),
    )
    assert searched.returncode == 0, searched.stderr
    assert "/cats.md[" in searched.stdout


@pytest.mark.parametrize("query,k", [
    ("", "1"), ("   ", "1"), ("[]", "1"), ("purring", "0"),
    ("purring", "100"), ("purring", "abc"),
])
def test_invalid_search_has_no_traceback(
    indexed_dir: Path, query: str, k: str
) -> None:
    """Turn invalid arguments into a nonzero exit and a readable message."""
    result = run_cli(
        "search", query, "--k", k, "--index_dir", str(indexed_dir),
    )
    assert result.returncode != 0
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr, result.stderr


def test_malformed_json_has_no_traceback(indexed_dir: Path) -> None:
    """Handle invalid JSON at the command boundary."""
    (indexed_dir / "chunks.jsonl").write_text("{broken json\n")
    result = run_cli(
        "search", "purring", "--k", "1", "--index_dir", str(indexed_dir),
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, result.stderr


def test_missing_index_has_no_traceback(tmp_path: Path) -> None:
    """Explain missing processed data without a Python traceback."""
    result = run_cli(
        "search", "purring", "--index_dir", str(tmp_path / "missing"),
    )
    assert result.returncode != 0
    assert "chunks.jsonl" in result.stderr
    assert "Traceback" not in result.stderr


def test_search_dataset_cli(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path
) -> None:
    """Produce a valid batch JSON through Fire, including paths with spaces."""
    output = tmp_path / "dataset results"
    result = run_cli(
        "search_dataset", "--dataset_path", str(dataset_path), "--k", "1",
        "--save_directory", str(output),
        "--index_directory", str(indexed_dir),
    )
    assert result.returncode == 0, result.stderr
    saved = StudentSearchResults.model_validate_json(
        (output / dataset_path.name).read_text(encoding="utf-8")
    )
    assert saved.k == 1
    assert [q.question_id for q in saved.search_results] == [
        "dog-id", "cat-id", "bird-id",
    ]
    assert [Path(q.retrieved_sources[0].file_path).name
            for q in saved.search_results] == [
        "dogs.md", "cats.md", "birds.py",
    ]


@pytest.mark.parametrize("text", [
    "{broken json", "{}", '{"rag_questions": [{"question": "   "}]}',
    '{"rag_questions": []}',
])
def test_invalid_dataset_cli_has_no_traceback(
    tmp_path: Path, indexed_dir: Path, text: str
) -> None:
    """Handle malformed JSON, wrong schemas, and blank questions in the CLI."""
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "results"
    result = run_cli(
        "search_dataset", "--dataset_path", str(path), "--k", "1",
        "--save_directory", str(output),
        "--index_directory", str(indexed_dir),
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, result.stderr
    assert not (output / path.name).exists()


@pytest.mark.parametrize("path", ["", "   "])
def test_blank_index_input_has_no_traceback(path: str) -> None:
    """Reject a blank corpus path before creating or overwriting an index."""
    result = run_cli("index", "--input_directory", path)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize("path", ["", "   "])
def test_blank_search_index_has_no_traceback(path: str) -> None:
    """Explain a blank index path without leaking a Python exception."""
    result = run_cli("search", "purring", "--k", "1", "--index_dir", path)
    assert result.returncode != 0
    assert "empty" in result.stderr
    assert "Traceback" not in result.stderr


def test_dataset_output_is_utf8_in_ascii_locale(
    dataset_path: Path, indexed_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save non-ASCII questions regardless of the process locale."""
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONCOERCECLOCALE", "0")
    output = tmp_path / "results"
    result = run_cli(
        "search_dataset", "--dataset_path", str(dataset_path), "--k", "1",
        "--save_directory", str(output),
        "--index_directory", str(indexed_dir),
    )
    assert result.returncode == 0, result.stderr
    saved = StudentSearchResults.model_validate_json(
        (output / dataset_path.name).read_text(encoding="utf-8"))
    assert saved.search_results[1].question == "Where is purring? 🐈"
