"""Exercise multiple AST nodes sharing a physical line."""

FIRST = 1; SECOND = 2  # noqa: E702
THIRD: int = 3; FOURTH: int = 4  # noqa: E702


def total() -> int:
    """Keep same-line statements inside a function."""
    left = FIRST; right = SECOND  # noqa: E702
    return left + right + THIRD + FOURTH


# Final comment after the last AST node.
