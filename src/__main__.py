import fire
from .index import IndexChunks
from .errors import IndexingException
from sys import stderr

if __name__ == "__main__":
    try:
        fire.Fire({
            "index": IndexChunks().index
        })
    except IndexingException as e:
        print(e, file=stderr)
