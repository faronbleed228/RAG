# Комментарий до первого AST-узла: он тоже должен сохраниться в чанках.
"""Provide source text for manually checking Python chunk boundaries.

This file uses only the standard library and is compatible with Python 3.10.
Parse its source without importing or executing it. Check consecutive imports,
decorators, nested blocks, Unicode, oversized nodes, and the final file tail.
The function long_report is intentionally longer than 2000 source characters.
"""

import asyncio
import json
from dataclasses import dataclass
from functools import cache
from functools import lru_cache


# Соседние присваивания и разные виды AST-узлов.
DEFAULT_LIMIT = 5
DEFAULT_PREFIX = "report"
GREETING: str = "Привет, мир! 🐍"
EMPTY_TEXT: str = ""

CONFIG: dict[str, object] = {
    "limit": DEFAULT_LIMIT,
    "prefix": DEFAULT_PREFIX,
    "greeting": GREETING,
    "enabled": True,
}


def add(left: int, right: int) -> int:
    """Return the sum of two integers."""
    return left + right


def subtract(left: int, right: int) -> int:
    """Keep a second adjacent FunctionDef separate from the first."""
    return left - right


# Оба декоратора должны оставаться вместе с функцией.
@cache
@lru_cache(maxsize=16)
def decorated_label(value: int) -> str:
    """Exercise stacked decorators and non-ASCII source characters."""
    return f"Результат: {value} ✓"


@lru_cache(
    maxsize=32,
    typed=True,
)
def multiline_decorator(value: int) -> int:
    """Keep every line of a decorator call with its function."""
    return value * value


@dataclass(
    slots=True,
)
class Record:
    """Exercise a decorated class containing several nested AST nodes."""

    name: str
    values: list[int]

    @property
    def total(self) -> int:
        """Return the sum without dropping the property decorator."""
        return sum(self.values)

    def describe(self) -> str:
        """Preserve blank lines and comments inside a method."""
        # Этот комментарий относится к методу, а не к следующему классу.
        label = self.name.strip()

        if not label:
            label = "Без имени"

        return f"{label}: {self.total}"


class EmptyRecord:
    """Keep a second adjacent ClassDef separate from the first."""

    pass


async def async_total(values: list[int]) -> int:
    """Exercise AsyncFunctionDef and an await expression."""
    await asyncio.sleep(0)
    return sum(values)


async def async_label(value: int) -> str:
    """Exercise two adjacent async functions."""
    await asyncio.sleep(0)
    return decorated_label(value)


def with_nested_function(values: list[int]) -> list[int]:
    """Keep an inner function inside its outer function's source span."""

    def double(value: int) -> int:
        """Return twice the supplied value."""
        return value * 2

    return [double(value) for value in values if value >= 0]


def long_report(values: list[int], limit: int = DEFAULT_LIMIT) -> str:
    """Build a report with several possible internal chunk boundaries.

    This deliberately long docstring belongs to the function. Splitting only
    at top-level AST boundaries leaves the whole function in one large chunk.
    A smaller chunk limit therefore requires a second splitting strategy.
    Its output still needs to reference the original text exactly, including
    indentation, comments, punctuation, empty lines, and Unicode characters.

    The first stage copies the input values and normalizes the requested
    limit. The second stage separates negative values from nonnegative values.
    The next stage calculates totals and builds labels for individual items.
    Later stages exercise a while loop and a try-except-finally block. These
    statements are indented because they belong to a function, so searching
    for a line that starts literally with 'for' will not find the for loop.

    Character positions describe the source file, rather than the runtime
    values of its strings. For example, a Unicode character occupies one
    Python string position even when its UTF-8 encoding occupies several
    bytes. Reading this file as text and checking source[start:end] avoids
    confusing byte offsets with character offsets.

    After splitting this function, concatenate the resulting fragments in
    their original order. They should reproduce the complete source without
    any missing lines or repeated sections. Each fragment should be nonempty
    and no longer than the configured limit. A following top-level function
    makes the end of this oversized block visible to a top-level AST walk.
    """
    copied_values = list(values)
    normalized_limit = max(0, limit)
    selected_values = copied_values[:normalized_limit]
    positive_values: list[int] = []
    negative_values: list[int] = []
    labels: list[str] = []

    # Первая возможная граница внутри длинной функции.
    for value in selected_values:
        if value < 0:
            negative_values.append(value)
        else:
            positive_values.append(value)

    if positive_values:
        positive_total = sum(positive_values)
        positive_average = positive_total / len(positive_values)
        labels.append(f"Положительные: {positive_total}")
        labels.append(f"Среднее: {positive_average:.2f}")
    else:
        positive_total = 0
        positive_average = 0.0
        labels.append("Нет неотрицательных значений")

    if negative_values:
        negative_total = sum(negative_values)
        labels.append(f"Отрицательные: {negative_total}")
    else:
        negative_total = 0
        labels.append("Нет отрицательных значений")

    cursor = 0
    while cursor < len(selected_values):
        value = selected_values[cursor]
        labels.append(f"Элемент {cursor}: {value}")
        cursor += 1

    try:
        payload = {
            "values": selected_values,
            "positive_total": positive_total,
            "negative_total": negative_total,
            "positive_average": positive_average,
            "labels": labels,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        serialized = f"Ошибка сериализации: {error}"
    finally:
        labels.append("Обработка завершена")

    return serialized


def after_long_report() -> str:
    """Mark the boundary immediately after the oversized function."""
    return "This function must not be duplicated or lost."


# Разные AST-узлы на одной строке: не создавай между ними пустой чанк.
SAME_LINE_FIRST = 1; SAME_LINE_SECOND: int = 2  # noqa: E702

if DEFAULT_LIMIT > 0:
    STATUS = "enabled"
else:
    STATUS = "disabled"


def final_function() -> str:
    """Exercise saving the final group without a following AST node."""
    return "Конец файла ✓"


# Хвост после последнего AST-узла тоже должен попасть в результат.
