# Chunking fixtures

Read these files as UTF-8 source text; do not execute or import them.
Small limits such as 120 or 200 expose splitting paths; 2000 checks the
assignment's maximum. Some files deliberately contain unusual formatting.

| File | Case |
| --- | --- |
| `chunking_sample.py` | Existing broad Python sample with oversized blocks |
| `small_chunk_sample.py` | Existing small class and function sample |
| `unicode_offsets.py` | Unicode, combining accents, leading and trailing comments |
| `decorated_blocks.py` | Stacked decorators, adjacent functions, async and nested definitions |
| `same_line_nodes.py` | Multiple AST statements on the same line |
| `comments_only.py` | Nonempty Python source with an empty AST body |
| `empty.py`, `empty.md` | Zero-byte input |
| `whitespace_only.py` | Only spaces, tabs, and blank lines |
| `no_final_newline.py`, `no_final_newline.md` | Final character is not a newline |
| `long_line.py` | A source line longer than 2000 characters, followed by more code |
| `markdown_structure.md` | Headings, paragraphs, lists, quotes, table, and code |
| `markdown_fences.md` | Triple and quadruple backticks, tilde fences, nested fences |
| `markdown_unclosed_fence.md` | A code fence open at EOF |
| `long_paragraph.md` | A paragraph over 2000 characters without line breaks |
| `boundary_2000.md` | Exactly 2000 characters without a final newline |
| `boundary_2001.md` | Exactly 2001 characters without a final newline |
| `invalid_syntax.py.txt` | Malformed Python, excluded from normal `*.py` corpus scans |

For every supported input, check that chunks are nonempty, do not exceed
the chosen limit, and match their original character ranges. Concatenating
chunks for one file in their original order should reproduce the entire file.
Empty files may produce no chunks. Comments-only and whitespace-only files
still contain source characters: decide explicitly whether to preserve them
or skip them, and test that policy.

Advanced fence cases describe useful Markdown behavior; the current parser
may need improvements to recognize every delimiter. When testing malformed
Python handling, copy `invalid_syntax.py.txt` to a temporary `.py` file.
