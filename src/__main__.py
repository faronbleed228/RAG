import fire
from .errors import IndexingException, ChunkingException
from .indexing.index import index_data
from sys import stderr, exit
from pydantic import ValidationError
import pathlib


if __name__ == "__main__":
    try:
        fire.Fire({
            "index": index_data
        })
    except (IndexingException, AssertionError,
            OSError, ValidationError, ChunkingException) as e:
        print(e, file=stderr)
        exit(1)
    except KeyboardInterrupt:
        print("exit........", file=stderr)
        exit(1)
