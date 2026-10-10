# Кириллица и эмодзи до первого AST-узла: 🐍 🚀.
"""Exercise character offsets with non-ASCII source text."""

GREETING = "Привет, мир! 🌍"
ACCENTED = "café, naïve, Straße"
COMBINING = "е́"  # The accent is a separate Unicode code point.


def describe(name: str) -> str:
    """Keep Unicode inside a function and its source span."""
    return f"Здравствуй, {name}! ✓"


# Хвост файла тоже должен сохраниться: конец 🏁.
