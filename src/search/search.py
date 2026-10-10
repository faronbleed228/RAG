import bm25s
from pathlib import Path
from ..errors import SearchException
from ..indexing.chunker import DataChunk
from typing import Any


def search(query: str | list[str], k: int = 10,
           index_dir: str | None = None) -> None:
    try:
        if index_dir is None:
            chunk_path = Path(__file__).parent.parent.parent.resolve() / \
                "data" / "processed" / "chunks.jsonl"
            index_path = Path(__file__).parent.parent.parent.resolve() / \
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
    except TypeError as e:
        raise SearchException(
            "Wrong value type "
            f"for processed data directory: {type(index_dir)}") from e
    chunk_list: list[DataChunk] = load_chunks(chunk_path)
    retriever = bm25s.BM25.load(index_path)
    query_tokens = bm25s.tokenize(
        query, stopwords=[], leave=False, return_ids=True)
    documents, scores = retriever.retrieve(
        query_tokens, k=k, leave_progress=False)
    for document in documents:
        for index in document:
            path = chunk_list[index].min_source.file_path
            start = chunk_list[index].min_source.first_character_index
            end = chunk_list[index].min_source.last_character_index
            print(f"{path}[{start}:{end}]")


def load_chunks(chunks_path: Path) -> list[DataChunk]:

    with chunks_path.open(encoding="utf-8") as fd:
        chunks_list: list[DataChunk] = [
            DataChunk.model_validate_json(line) for line in fd if line.strip()]
    return chunks_list
