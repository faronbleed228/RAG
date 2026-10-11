import fire
from sys import stderr, exit
from pydantic import ValidationError
from .indexing.index import index_data
from .errors import IndexingException, ChunkingException, SearchException
from .search.search import search_cli
from .search.search_dataset import search_dataset

if __name__ == "__main__":
    try:
        fire.Fire({
            "index": index_data,
            "search": search_cli,
            "search_dataset": search_dataset

        })
    except KeyboardInterrupt:
        print("exit........", file=stderr)
        exit(1)
    except (IndexingException, AssertionError,
            OSError, ValidationError, ChunkingException,
            SearchException) as e:
        print(e, file=stderr)
        exit(1)
