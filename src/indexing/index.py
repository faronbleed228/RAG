from .chunker import BaseChunk, DataChunk, MinimalSource
from pathlib import Path
import bm25s
from pydantic import TypeAdapter


def index_data(max_chunk_size: int = 2000) -> None:
    chunker = BaseChunk(max_chunk_size)
    chunk_list: list[DataChunk] = chunker.chunk_loop()
    # print_chunks_info(chunk_list)
    load_chunks(chunk_list)
    create_search_index(chunk_list)


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
    print(new_chunk_list == chunk_list)


def create_search_index(chunk_list: list[DataChunk]) -> None:
    data_path = Path(__file__).resolve(
    ).parent.parent.parent / "data" / "processed" / "bm25"

    data_path.mkdir(parents=True, exist_ok=True)

    # stemmer = Stemmer.Stemmer("english")
    texts: list[str] = [chunk.content for chunk in chunk_list]
    corpus_tokens = bm25s.tokenize(texts, stopwords=[], return_ids=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(data_path)
