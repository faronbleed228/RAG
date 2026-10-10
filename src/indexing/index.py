from .chunker import BaseChunk, DataChunk
from pathlib import Path
from ..errors import IndexingException
import bm25s
from pydantic import TypeAdapter


def index_data(max_chunk_size: int = 2000,
               input_directory: str | None = None,
               index_directory: str | None = None) -> None:

    chunker = BaseChunk(max_chunk_size)
    chunker.update_path(input_directory)
    chunk_list: list[DataChunk] = chunker.chunk_loop()

    # print_chunks_info(chunk_list)
    load_chunks(chunk_list)
    create_search_index(chunk_list, index_directory)


def print_chunks_info(chunk_list: list[DataChunk]) -> None:
    for chunk in chunk_list:
        print("Chunk index", chunk.index)
        print("Chunk first char", chunk.min_source.first_character_index)
        print("Chunk last char", chunk.min_source.last_character_index)
        print()


def load_chunks(chunk_list: list[DataChunk]) -> None:
    data_path = Path(__file__).resolve(
    ).parent.parent.parent / "data" / "processed" / "chunks.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as output:
        for chunk in chunk_list:
            output.write(chunk.model_dump_json() + "\n")
    new_chunk_list: list[DataChunk] = []
    with data_path.open("r", encoding="utf_8") as input:
        for line in input:
            if line.strip():
                chunk = DataChunk.model_validate_json(line)
                new_chunk_list.append(chunk)


def create_search_index(chunk_list: list[DataChunk],
                        index_directory: str | None) -> None:
    try:
        if index_directory is None:
            data_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "processed" / "bm25"
        else:
            data_path = Path(index_directory).resolve()
    except TypeError as e:
        raise IndexingException(
            "Wrong value type "
            f"for index_directory: {type(index_directory)}") from e

    data_path.mkdir(parents=True, exist_ok=True)

    texts: list[str] = [chunk.content for chunk in chunk_list]
    if len(texts) == 0:
        raise IndexingException("All data files are empty")
    corpus_tokens = bm25s.tokenize(texts, stopwords=[], return_ids=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(data_path)
