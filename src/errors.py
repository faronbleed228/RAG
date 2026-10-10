class IndexingException(Exception):
    def __init__(self, msg: str = "Default indexing error: "
                 "something went wrong") -> None:
        super().__init__(msg)


class ChunkingException(Exception):
    def __init__(self,
                 msg: str = "Default chunking error:"
                 " something went wrong") -> None:
        super().__init__(msg)


class SearchException(Exception):
    def __init__(self,
                 msg: str = "Default search error:"
                 " something went wrong") -> None:
        super().__init__(msg)
