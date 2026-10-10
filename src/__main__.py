import fire
from .errors import IndexingException
from .indexing.index import index_data
from sys import stderr
from pydantic import ValidationError


if __name__ == "__main__":
    try:
        fire.Fire({
            "index": index_data
        })
    except (IndexingException, AssertionError, OSError, KeyboardInterrupt, ValidationError) as e:
        print(e, file=stderr)
