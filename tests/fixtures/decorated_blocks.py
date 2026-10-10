"""Exercise decorators, adjacent definitions, and nested blocks."""

from functools import cache, lru_cache


@cache
@lru_cache(
    maxsize=8,
    typed=True,
)
def square(value: int) -> int:
    """Keep both decorators with this function when they fit."""
    return value * value


def cube(value: int) -> int:
    """Keep an adjacent function distinguishable."""
    return value * value * value


class Container:
    """Exercise method boundaries when using a small chunk limit."""

    @property
    def label(self) -> str:
        """Keep the property decorator attached to its method."""
        return "container"

    @classmethod
    def create(cls) -> "Container":
        """Exercise a class method."""
        return cls()

    async def read(self) -> str:
        """Exercise an async method inside a class."""
        return self.label


async def nested(value: int) -> int:
    """Keep a nested function within an async function."""

    def double(number: int) -> int:
        """Return twice the input."""
        return number * 2

    return double(value)
