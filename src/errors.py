class IndexingException(Exception):
    def __init__(self, msg: str = "Default error: something went wrong") -> None:
        super().__init__(msg)
