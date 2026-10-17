"""Check persisted chunks and their mapping to a real BM25 index."""

import json
from pathlib import Path

import bm25s
import pytest

from src.errors import IndexingException
from src.indexing.chunker import BaseChunk
from src.indexing.index import create_search_index, index_data, load_chunks
from src.search.search import _load_chunks as read_chunks


def test_chunks_jsonl_round_trip(corpus_dir: Path, tmp_path: Path) -> None:
    """Serialize validated chunks without changing their order or content."""
    chunker = BaseChunk(2000)
    chunker.update_path(str(corpus_dir))
    chunks = chunker.chunk_loop()
    directory = tmp_path / "nested" / "processed"
    load_chunks(chunks, str(directory))

    path = directory / "chunks.jsonl"
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(chunks)
    assert read_chunks(path, k=1) == chunks


def test_saved_index_matches_chunks(indexed_dir: Path) -> None:
    """Keep BM25 document IDs aligned with JSONL source locations."""
    chunks = read_chunks(indexed_dir / "chunks.jsonl", k=1)
    params = json.loads(
        (indexed_dir / "bm25" / "params.index.json").read_text()
    )
    assert params["num_docs"] == len(chunks)
    retriever = bm25s.BM25.load(indexed_dir / "bm25")
    tokens = bm25s.tokenize("purring", stopwords=[], show_progress=False)
    documents, scores = retriever.retrieve(tokens, k=1, show_progress=False)
    chunk = chunks[int(documents[0, 0])]

    assert scores[0, 0] > 0
    assert chunk.min_source.file_path.endswith("/cats.md")
    assert "purring" in chunk.content


def test_index_rejects_empty_chunks(tmp_path: Path) -> None:
    """Reject an empty tokenization corpus with a domain exception."""
    with pytest.raises(IndexingException, match="empty"):
        create_search_index([], str(tmp_path))


def test_empty_reindex_preserves_saved_index(
    corpus_dir: Path, indexed_dir: Path
) -> None:
    """Keep every saved file intact when a subsequent corpus is empty."""
    before = {
        path.relative_to(indexed_dir): path.read_bytes()
        for path in indexed_dir.rglob("*") if path.is_file()
    }
    for path in corpus_dir.iterdir():
        path.write_text("", encoding="utf-8")

    with pytest.raises(IndexingException, match="empty"):
        index_data(
            input_directory=str(corpus_dir),
            index_directory=str(indexed_dir),
        )

    after = {
        path.relative_to(indexed_dir): path.read_bytes()
        for path in indexed_dir.rglob("*") if path.is_file()
    }
    assert after == before


def _snapshot(directory: Path) -> dict[str, bytes]:
    """Read all stored artifacts for comparison after a failed rebuild."""
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("content", [
    "!!! ??? ...", "   \n", "what which who does do did",
])
def test_empty_vocabulary_preserves_saved_index(
    corpus_dir: Path, indexed_dir: Path, content: str,
) -> None:
    """Reject nonempty text without tokens before replacing saved data."""
    before = _snapshot(indexed_dir)
    for path in corpus_dir.iterdir():
        path.unlink()
    (corpus_dir / "only.txt").write_text(content, encoding="utf-8")

    with pytest.raises(IndexingException, match="empty vocabulary"):
        index_data(input_directory=str(corpus_dir),
                   index_directory=str(indexed_dir))
    assert _snapshot(indexed_dir) == before


def test_failed_bm25_save_preserves_saved_index(
    corpus_dir: Path, indexed_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the old pair when writing the staged BM25 artifacts fails."""
    before = _snapshot(indexed_dir)

    def fail_save(*args: object, **kwargs: object) -> None:
        """Simulate a disk error during serialization."""
        raise OSError("simulated save failure")

    monkeypatch.setattr(bm25s.BM25, "save", fail_save)
    with pytest.raises(OSError, match="simulated save failure"):
        index_data(input_directory=str(corpus_dir),
                   index_directory=str(indexed_dir))
    assert _snapshot(indexed_dir) == before


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_publish_failure_restores_both_artifacts(
    corpus_dir: Path, indexed_dir: Path, monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """Restore both old artifacts if publication stops after replacing one."""
    before = _snapshot(indexed_dir)
    original_replace = Path.replace

    def fail_replace(path: Path, target: str | Path) -> Path:
        """Fail only while installing the new BM25 directory."""
        if (path.name == "bm25"
                and path.parent.name.startswith(".rag-index-")
                and Path(target) == indexed_dir / "bm25"):
            raise failure("simulated publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(failure, match="simulated publish failure"):
        index_data(input_directory=str(corpus_dir),
                   index_directory=str(indexed_dir))
    assert _snapshot(indexed_dir) == before


def test_successful_reindex_preserves_unrelated_files(
    corpus_dir: Path, indexed_dir: Path,
) -> None:
    """Publish the new matching pair without replacing the whole directory."""
    sentinel = indexed_dir / "keep.txt"
    sentinel.write_text("unrelated file", encoding="utf-8")
    (corpus_dir / "cats.md").write_text("newuniqueword", encoding="utf-8")
    index_data(input_directory=str(corpus_dir),
               index_directory=str(indexed_dir))
    chunks = read_chunks(indexed_dir / "chunks.jsonl", k=1)
    retriever = bm25s.BM25.load(indexed_dir / "bm25")
    tokens = bm25s.tokenize(
        "newuniqueword", return_ids=False, show_progress=False)
    documents, _ = retriever.retrieve(tokens, k=1, show_progress=False)
    assert chunks[int(documents[0, 0])].content == "newuniqueword"
    assert retriever.scores["num_docs"] == len(chunks)
    assert sentinel.read_text(encoding="utf-8") == "unrelated file"
