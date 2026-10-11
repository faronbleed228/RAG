"""Exercise ranking, source ranges, and invalid search inputs."""

from pathlib import Path
import re
from typing import Any

import pytest
from pydantic import ValidationError

from src.errors import SearchException
from src.search.search import _load_chunks, search, search_cli


@pytest.mark.parametrize("k", [1, 3])
def test_search_returns_ranked_source_spans(
    indexed_dir: Path, capsys: pytest.CaptureFixture[str], k: int
) -> None:
    """Find the distinctive word and print valid source character ranges."""
    capsys.readouterr()
    search_cli("purring", k=k, index_dir=str(indexed_dir))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == k
    assert "/cats.md[" in lines[0]

    project_root = Path(__file__).resolve().parent.parent
    contents = []
    for line in lines:
        match = re.fullmatch(r"(.+)\[(\d+):(\d+)\]", line)
        assert match is not None, line
        path, first, last = match.groups()
        source = (project_root / path).read_text(encoding="utf-8")
        start, end = int(first), int(last)
        assert 0 <= start < end <= len(source)
        assert end - start <= 2000
        contents.append(source[start:end])
    assert "purring" in contents[0]


@pytest.mark.parametrize("query", [
    "", " \t\n", None, 123, {}, ["purring", ""], ["purring", 123],
])
def test_invalid_query(indexed_dir: Path, query: Any) -> None:
    """Reject blank text and non-string query values."""
    with pytest.raises(SearchException, match="query"):
        search(query, k=1, index_dir=str(indexed_dir))


@pytest.mark.parametrize(
    "k", [0, -1, 101, 10000, True, False, 1.5, 1.0, "abc", None]
)
def test_invalid_k(indexed_dir: Path, k: Any) -> None:
    """Reject invalid counts without leaking a library or type error."""
    with pytest.raises(SearchException, match="--k"):
        search("purring", k=k, index_dir=str(indexed_dir))


def test_k_above_corpus_size(indexed_dir: Path) -> None:
    """Reject a count larger than this index even if it is below 100."""
    count = len(_load_chunks(indexed_dir / "chunks.jsonl", k=1))
    with pytest.raises(SearchException, match="--k"):
        search("purring", k=count + 1, index_dir=str(indexed_dir))


def test_missing_index_directory(tmp_path: Path) -> None:
    """Explain when no saved chunks are available."""
    with pytest.raises(SearchException, match="chunks.jsonl"):
        search("purring", k=1, index_dir=str(tmp_path / "missing"))


def test_missing_bm25_directory(tmp_path: Path) -> None:
    """Check for the index directory before trying to load its arrays."""
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SearchException, match="bm25"):
        search("purring", k=1, index_dir=str(tmp_path))


@pytest.mark.parametrize("filename", ["vocab.index.json",
                                      "params.index.json",
                                      "data.csc.index.npy"])
def test_corrupt_index(indexed_dir: Path, filename: str) -> None:
    """Wrap real JSON and NumPy deserialization failures."""
    (indexed_dir / "bm25" / filename).write_bytes(b"{broken file")
    with pytest.raises(SearchException, match="load index") as error:
        search("purring", k=1, index_dir=str(indexed_dir))
    assert isinstance(error.value.__cause__, (OSError, ValueError, EOFError))


def test_missing_index_array(indexed_dir: Path) -> None:
    """Wrap a missing file inside an otherwise existing index directory."""
    (indexed_dir / "bm25" / "data.csc.index.npy").unlink()
    with pytest.raises(SearchException, match="load index") as error:
        search("purring", k=1, index_dir=str(indexed_dir))
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_malformed_chunks_json(tmp_path: Path) -> None:
    """Reject malformed JSON rather than silently dropping a source."""
    path = tmp_path / "chunks.jsonl"
    path.write_text("{broken json\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        _load_chunks(path, k=1)


def test_chunk_loader_ignores_blank_lines(indexed_dir: Path) -> None:
    """Allow blank lines while retaining chunk order and metadata."""
    path = indexed_dir / "chunks.jsonl"
    expected = _load_chunks(path, k=1)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n \t\n" + text + "\n", encoding="utf-8")
    assert _load_chunks(path, k=1) == expected


@pytest.mark.parametrize("k", [1, 3])
def test_search_returns_data_without_printing(
    indexed_dir: Path, capsys: pytest.CaptureFixture[str], k: int
) -> None:
    """Return one ranking per query without mixing CLI output into data."""
    capsys.readouterr()
    chunks, documents = search(
        ["barking", "purring"], k=k, index_dir=str(indexed_dir),
    )
    assert capsys.readouterr().out == ""
    assert documents.shape == (2, k)
    for row, filename in zip(documents, ["dogs.md", "cats.md"], strict=True):
        source = chunks[int(row[0])].min_source
        assert source.file_path.endswith("/" + filename)
    for row in documents:
        assert all(0 <= int(index) < len(chunks) for index in row)


def test_batch_matches_individual_searches(indexed_dir: Path) -> None:
    """Preserve each query's ranking when several queries are sent together."""
    queries = ["chirping", "purring", "barking"]
    chunks, documents = search(queries, k=1, index_dir=str(indexed_dir))
    for query, row in zip(queries, documents, strict=True):
        single_chunks, single_documents = search(
            query, k=1, index_dir=str(indexed_dir),
        )
        batch_source = chunks[int(row[0])].min_source
        single_source = single_chunks[int(single_documents[0, 0])].min_source
        assert batch_source == single_source
