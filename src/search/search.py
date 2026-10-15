import bm25s
from pathlib import Path
from ..errors import SearchException
from ..indexing.chunker import DataChunk
from ..indexing.index import get_more_token
from typing import Any


def search_cli(query: str, k: int = 10,
               index_dir: str | None = None) -> None:
    chunk_list, documents = search(query, k, index_dir)
    for document in documents:
        for index in document:
            path = chunk_list[index].min_source.file_path
            start = chunk_list[index].min_source.first_character_index
            end = chunk_list[index].min_source.last_character_index
            print(f"{path}[{start}:{end}]")


def search(query: str | list[str], k: int,
           index_dir: str | None) -> tuple[list[DataChunk], Any]:
    index_path, chunk_path = _check_input(query, index_dir)

    chunk_list: list[DataChunk] = _load_chunks(chunk_path, k)

    try:
        retriever = bm25s.BM25.load(index_path)
    except (OSError, ValueError, EOFError) as e:
        raise SearchException(
            f"Couldnt load index from {index_path}. "
            "Check files and try again"
        ) from e
    try:
        if isinstance(query, list):
            query_list = [get_more_token(q) for q in query]
        else:
            query_list = get_more_token(query)
        query_tokens = bm25s.tokenize(
            query_list, stopwords="english",
            leave=False, return_ids=False, lower=True)
        documents, scores = retriever.retrieve(
            query_tokens, k=k, leave_progress=False)
    except (TypeError, AttributeError) as e:
        raise SearchException(
            f"Couldnt retrieve indexes from {index_path}. "
            "Check files and try again"
        ) from e

    return (chunk_list, documents)


def _load_chunks(chunks_path: Path, k: int) -> list[DataChunk]:
    with chunks_path.open(encoding="utf-8") as fd:
        chunk_list: list[DataChunk] = [
            DataChunk.model_validate_json(line) for line in fd if line.strip()]

    if not isinstance(k, int) or isinstance(k, bool):
        raise SearchException("--k parameter has to be an int")
    if k not in range(1, 101) or k > len(chunk_list):
        raise SearchException(
            "--k parameter has to be in range 1-100 and "
            f"at less than amount of chunks({len(chunk_list)})")

    return chunk_list


def _check_input(query: str | list[str],
                 index_dir: str | None) -> tuple[Path, Path]:
    if isinstance(query, list):
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
