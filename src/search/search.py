"""Load persisted chunks and rank them with BM25 for one or more queries."""

import bm25s
from pathlib import Path
from ..errors import SearchException
from ..indexing.chunker import DataChunk
from ..indexing.index import get_more_token
from typing import Any


def search_cli(query: str, k: int = 10,
               index_dir: str | None = None) -> None:
    """Print the top-k source paths and exclusive character ranges.

    Args:
        query: Nonblank query text.
        k: Number of results, from 1 to 100 and no larger than the corpus.
        index_dir: Directory containing chunks.jsonl and bm25; None uses
            data/processed under the project root.
    """
    chunk_list, documents = search(query, k, index_dir)
    for document in documents:
        for index in document:
            path = chunk_list[index].min_source.file_path
            start = chunk_list[index].min_source.first_character_index
            end = chunk_list[index].min_source.last_character_index
            print(f"{path}[{start}:{end}]")


def search(query: str | list[str], k: int,
           index_dir: str | None) -> tuple[list[DataChunk], Any]:
    """Rank chunks for one query or a batch, loading the index once.

    Args:
        query: Nonblank text or a list of nonblank query strings.
        k: Number of results per query, limited to 1-100 and the chunk count.
        index_dir: Index directory, or None for data/processed.

    Returns:
        The loaded chunks and an array of their positional IDs with shape
        (number of queries, k), ordered by decreasing BM25 score per row.

    Raises:
        SearchException: If queries, k or the persisted index are invalid.
    """
    index_path, chunk_path = _check_input(query, index_dir)

    chunk_list: list[DataChunk] = _load_chunks(chunk_path, k)

    try:
        retriever = bm25s.BM25.load(index_path)
        if retriever.scores["num_docs"] != len(chunk_list):
            raise SearchException("Index and chunk counts do not match; "
                                  "rebuild the index")
    except (OSError, ValueError, EOFError, AttributeError,
            TypeError, KeyError, IndexError) as e:
        raise SearchException(
            f"Couldnt load index from {index_path}. "
            "Check files and try again"
        ) from e
    try:
        query_list: str | list[str]
        if isinstance(query, list):
            query_list = [get_more_token(q) for q in query]
        else:
            query_list = get_more_token(query)
        query_tokens = bm25s.tokenize(
            query_list, stopwords="english",
            leave=False, return_ids=False, lower=True)
        documents, scores = retriever.retrieve(
            query_tokens, k=k, leave_progress=False)
        if any(index < 0 or index >= len(chunk_list)
               for document in documents for index in document):
            raise SearchException("Index returned an invalid chunk ID; "
                                  "rebuild the index")
    except (TypeError, AttributeError, ValueError, KeyError, IndexError) as e:
        raise SearchException(
            f"Couldnt retrieve indexes from {index_path}. "
            "Check files and try again"
        ) from e

    return (chunk_list, documents)


def _load_chunks(chunks_path: Path, k: int) -> list[DataChunk]:
    """Read nonblank JSONL records in order and validate k against them."""
    with chunks_path.open(encoding="utf-8") as fd:
        chunk_list: list[DataChunk] = [
            DataChunk.model_validate_json(line) for line in fd if line.strip()]

    if any(chunk.index != index for index, chunk in enumerate(chunk_list)):
        raise SearchException("Chunk IDs are not in index order; "
                              "rebuild the index")

    if not isinstance(k, int) or isinstance(k, bool):
        raise SearchException("--k parameter has to be an int")
    if k not in range(1, 101) or k > len(chunk_list):
        raise SearchException(
            "--k parameter has to be in range 1-100 and "
            f"at less than amount of chunks({len(chunk_list)})")

    return chunk_list


def _check_input(query: str | list[str],
                 index_dir: str | None) -> tuple[Path, Path]:
    """Check query values and return the existing BM25 and JSONL paths."""
    if isinstance(query, list):
        if not query:
            raise SearchException("Empty query list")
        for q in query:
            if not isinstance(q, str) or q.strip() == "":
                raise SearchException(
                    "Not valid query type (has to be a valid string)")
    elif not isinstance(query, str) or query.strip() == "":
        raise SearchException(
            "Not valid query type (has to be a valid string)")

    try:
        if index_dir is None:
            chunk_path = Path(__file__).resolve().parent.parent.parent / \
                "data" / "processed" / "chunks.jsonl"
            index_path = Path(__file__).resolve().parent.parent.parent / \
                "data" / "processed" / "bm25"
        else:
            if index_dir.strip() == "":
                raise SearchException("Index_dir path should not be empty")
            chunk_path = Path(index_dir, "chunks.jsonl").resolve()
            index_path = Path(index_dir, "bm25").resolve()
        if not chunk_path.is_file() or not chunk_path.exists():
            raise SearchException(
                "Could not resolve chunks.jsonl file. "
                "Either indexing wasnt done yet or"
                " save directory was changed.")
        if not index_path.is_dir() or not index_path.exists():
            raise SearchException(
                "Could not resolve bm25 folder. "
                "Either indexing wasnt done yet or"
                " save directory was changed.")
    except (TypeError, AttributeError) as e:
        raise SearchException(
            "Wrong value type "
            f"for processed data directory: {type(index_dir)}") from e
    return (index_path, chunk_path)
