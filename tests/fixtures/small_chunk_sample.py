"""Small Python source for trying a max_chunk_size of about 120."""

BASE = 1


class Counter:
    """This class is longer than one small chunk."""

    def __init__(self) -> None:
        self.value = BASE

    def add(self, amount: int) -> int:
        self.value += amount
        return self.value


def label(counter: Counter) -> str:
    """Keep one function after the class."""
    return f"value={counter.value}"
