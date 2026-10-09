import fire
from .errors import IndexingException
from .indexing.index import index_data
from sys import stderr


if __name__ == "__main__":
    try:
        fire.Fire({
            "index": index_data()
        })
    except (IndexingException, AssertionError, OSError, KeyboardInterrupt) as e:
        print(e, file=stderr)
