from .chunker import BaseChunk, DataChunk, MinimalSource
from pathlib import Path
import bm25s


def index_data(max_chunk_size: int = 2000) -> None:
    chunker = BaseChunk(max_chunk_size)
    chunk_list: list[DataChunk] = chunker.chunk_loop()
    # print_chunks_info(chunk_list)
    load_chunks(chunk_list)


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


def create_search_index() -> None:
    data_path = Path(__file__).resolve(
    ).parent.parent.parent / "data" / "processed" / "chunks.jsonl"

