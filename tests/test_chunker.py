"""Check source preservation, character spans, and chunking failures."""

from collections import defaultdict
from pathlib import Path

import pytest

from src.errors import ChunkingException
from src.indexing.chunker import BaseChunk, DataChunk


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("limit", [10, 120, 200, 2000])
def test_chunker(limit: int) -> None:
    """Preserve every source, including the last file and Unicode offsets."""
    chunker = BaseChunk(limit)
    chunker.data_path = FIXTURES
    chunks = chunker.chunk_loop()
    grouped: dict[str, list[DataChunk]] = defaultdict(list)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        grouped[chunk.min_source.file_path].append(chunk)

    sources = {
        str(path.relative_to(chunker.absolute_path)): path
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".txt"}
    }
    assert set(grouped) <= set(sources)
    for relative_path, path in sources.items():
        source = path.read_text(encoding="utf-8")
        parts = grouped.get(relative_path, [])
        assert "".join(part.content for part in parts) == source, relative_path
        cursor = 0
        for part in parts:
            start = part.min_source.first_character_index
            end = part.min_source.last_character_index
            assert start == cursor, relative_path
            assert 0 <= start < end <= len(source), relative_path
            assert 0 < len(part.content) <= limit, relative_path
            assert source[start:end] == part.content, relative_path
            cursor = end


@pytest.mark.parametrize("limit", [-1, 0, 2001])
def test_invalid_chunk_limit(limit: int) -> None:
    """Reject invalid sizes before reading a corpus."""
    chunker = BaseChunk(limit)
    chunker.data_path = FIXTURES
    with pytest.raises(ChunkingException, match="max_chunk_size"):
        chunker.chunk_loop()


def test_missing_corpus(tmp_path: Path) -> None:
    """Report an absent corpus instead of silently producing no chunks."""
    chunker = BaseChunk(2000)
    chunker.data_path = tmp_path / "missing"
    with pytest.raises(ChunkingException, match="does not exist"):
        chunker.chunk_loop()


def test_external_corpus_is_rejected(tmp_path: Path) -> None:
    """Reject input outside the project with a domain exception."""
    chunker = BaseChunk(2000)
    with pytest.raises(ChunkingException, match="outside"):
        chunker.update_path(str(tmp_path))


@pytest.mark.parametrize("path", ["", " \t\n"])
def test_blank_input_path(path: str) -> None:
    """Reject explicit blank paths instead of indexing the current folder."""
    with pytest.raises(ChunkingException, match="empty"):
        BaseChunk(2000).update_path(path)


@pytest.mark.parametrize("suffix", [".py", ".md", ".txt"])
def test_empty_file(tmp_path: Path, suffix: str) -> None:
    """Do not emit zero-length chunks for an empty file."""
    (tmp_path / f"empty{suffix}").write_text("", encoding="utf-8")
    chunker = BaseChunk(2000)
    chunker.absolute_path = tmp_path
    chunker.data_path = tmp_path
    assert chunker.chunk_loop() == []


def test_empty_directory(tmp_path: Path) -> None:
    """Allow an empty corpus at the chunking stage."""
    chunker = BaseChunk(2000)
    chunker.data_path = tmp_path
    assert chunker.chunk_loop() == []


def test_directories_with_file_extensions(tmp_path: Path) -> None:
    """Ignore directories whose names match a supported file pattern."""
    for name in ("amd", "folder.md", "folder.py", "folder.txt"):
        (tmp_path / name).mkdir()
    chunker = BaseChunk(2000)
    chunker.data_path = tmp_path
    assert chunker.chunk_loop() == []


def test_invalid_python(tmp_path: Path) -> None:
    """Wrap syntax errors in a chunking exception."""
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    chunker = BaseChunk(2000)
    chunker.absolute_path = tmp_path
    chunker.data_path = tmp_path
    with pytest.raises(ChunkingException, match="broken.py") as error:
        chunker.chunk_loop()
    assert isinstance(error.value.__cause__, SyntaxError)


@pytest.mark.parametrize("suffix", [".py", ".md", ".txt"])
def test_invalid_utf8(tmp_path: Path, suffix: str) -> None:
    """Wrap decoding errors for every supported file format."""
    (tmp_path / f"broken{suffix}").write_bytes(b"\xff\xfe\xfa")
    chunker = BaseChunk(2000)
    chunker.data_path = tmp_path
    with pytest.raises(ChunkingException) as error:
        chunker.chunk_loop()
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_small_decorated_function_stays_together(tmp_path: Path) -> None:
    """Preserve stacked decorators with a function that fits the limit."""
    source = (
        "from functools import cache\n\n\n"
        "@cache\n@cache\ndef square(value: int) -> int:\n"
        '    """Return a square."""\n    return value * value\n'
    )
    (tmp_path / "decorators.py").write_text(source, encoding="utf-8")
    chunker = BaseChunk(2000)
    chunker.absolute_path = tmp_path
    chunker.data_path = tmp_path
    chunks = chunker.chunk_loop()
    decorated = source[source.index("@cache"):]
    assert any(decorated in chunk.content for chunk in chunks)
