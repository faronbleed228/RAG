"""Define errors reported by indexing, retrieval and dataset commands."""


class IndexingException(Exception):
    """Report an index construction, persistence or publication error."""

    def __init__(self, msg: str = "Default indexing error: "
                 "something went wrong") -> None:
        """Initialize the error with a message suitable for CLI output."""
        super().__init__(msg)


class ChunkingException(Exception):
    """Report invalid chunking inputs or unreadable source text."""

    def __init__(self,
                 msg: str = "Default chunking error:"
                 " something went wrong") -> None:
        """Initialize the error with a message suitable for CLI output."""
        super().__init__(msg)


class SearchException(Exception):
    """Report a retrieval, answer dataset or evaluation error."""

    def __init__(self,
                 msg: str = "Default search error:"
                 " something went wrong") -> None:
        """Initialize the error with a message suitable for CLI output."""
        super().__init__(msg)
