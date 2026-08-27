"""Stage 1 - reading the input file.

Two things here are less trivial than they look, and both come from the задание's
requirement that the system accept *arbitrary* raw files rather than known ones.

**Encoding is declared, not assumed.** `pd.read_csv` defaults to UTF-8 and raises
on the first byte it cannot decode. Real files are often not UTF-8: a file
exported from Excel on a Western European machine is usually cp1252, where the
pound sign is byte 0xA3 -- invalid UTF-8. Guessing silently would be worse than
failing, so we try a short ordered list and *record which one worked*, so the
choice appears in the report rather than being buried.

**Delimiter is sniffed for CSV.** Semicolon-separated files are the norm in
locales where the comma is the decimal separator, which includes Bulgarian Excel.
A semicolon file read with a comma delimiter yields a single column containing
the whole row -- no error, just a useless table. We check the header line and
pick whichever candidate splits it into the most fields.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

#: Tried in order. UTF-8 first because it is correct and most common; cp1252 next
#: because it is what Excel produces on Western European systems; latin-1 last
#: because it decodes *any* byte sequence and therefore never fails -- which makes
#: it a usable fallback but a meaningless success, so it is reported when used.
ENCODINGS = ("utf-8", "cp1252", "latin-1")

#: Candidate CSV delimiters, most likely first.
DELIMITERS = (",", ";", "\t", "|")

#: How many lines to examine when looking for junk above the header.
PREAMBLE_SCAN_LINES = 20

#: The header is accepted only if this many following rows agree on the field count,
#: so a genuine one-column file is not mistaken for a preamble.
PREAMBLE_CONFIRM_ROWS = 3

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


@dataclass
class LoadResult:
    """A loaded table plus how it was read."""

    frame: pd.DataFrame
    name: str
    encoding: str | None = None
    delimiter: str | None = None
    had_header: bool = True
    decimal: str = "."
    skipped_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.shape

    def summary(self) -> dict[str, Any]:
        """Flat description, for the report and the UI."""
        rows, cols = self.frame.shape
        return {
            "file": self.name,
            "rows": rows,
            "columns": cols,
            "encoding": self.encoding or "n/a",
            "delimiter": {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(
                self.delimiter or "", self.delimiter or "n/a"
            ),
            "header_row": "yes" if self.had_header else "no (names generated)",
            "decimal": self.decimal,
            "skipped_rows": self.skipped_rows,
            "notes": self.notes,
        }


def sniff_delimiter(sample: str) -> str:
    """Pick the delimiter that splits the first line into the most fields.

    Preferred over `csv.Sniffer` alone, which raises on ambiguous input; this
    always returns something and falls back to a comma.
    """
    first = sample.splitlines()[0] if sample.splitlines() else ""
    if not first:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        counts = {d: first.count(d) for d in DELIMITERS}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","


def sniff_preamble(text: str, delimiter: str) -> int:
    """How many junk lines sit above the real header.

    Spreadsheet exports routinely begin with a title row and a blank row before the
    header: read naively, the first column is then named after the report title,
    every column becomes text, and the table is short by however many rows were
    consumed. Nothing raises.

    The rule is deliberately conservative. The real header is taken to be the first
    line whose field count matches the count that the following lines agree on. If
    the very first line already matches, there is no preamble -- which is the normal
    case and must stay free of false positives.
    """
    # Counted with csv.reader, not by splitting lines: a quoted value containing a
    # newline makes one logical row span two physical lines, and splitting on
    # newlines then miscounts every row after it. That bug consumed a real header.
    from io import StringIO

    reader = csv.reader(StringIO(text), delimiter=delimiter)
    rows = []
    for row in reader:
        rows.append(row)
        if len(rows) >= PREAMBLE_SCAN_LINES:
            break
    if len(rows) < PREAMBLE_CONFIRM_ROWS + 1:
        return 0
    counts = [len(r) for r in rows]
    body = counts[1:]
    if not body:
        return 0
    # The width the file settles on, taken as the commonest count below the top.
    settled = max(set(body), key=body.count)
    if settled <= 1:
        return 0
    for index, count in enumerate(counts):
        if count != settled:
            continue
        following = counts[index + 1 : index + 1 + PREAMBLE_CONFIRM_ROWS]
        if following and all(c == settled for c in following):
            return index
    return 0


def read_table(
    source: str | Path | BinaryIO,
    *,
    has_header: bool = True,
    encoding: str | None = None,
    delimiter: str | None = None,
    decimal: str = ".",
    skip_rows: int | None = None,
) -> LoadResult:
    """Read a CSV or Excel file into a frame, recording how it was read.

    Args:
        source: a path, or an open binary file object (Streamlit's uploader).
        has_header: whether the first row holds column names. When False,
            columns are named `column_1 ... column_n`; a wrong guess here is
            silent and destructive, so it is never inferred.
        encoding: force an encoding instead of trying `ENCODINGS`.
        delimiter: force a CSV delimiter instead of sniffing.
        decimal: the decimal separator. `","` for files where 1,5 means one and a
            half, which is the convention across most of Europe.
        skip_rows: junk lines above the header. When None, detected by
            :func:`sniff_preamble` and reported; pass 0 to disable.
    """
    name = getattr(source, "name", None) or str(source)
    suffix = Path(str(name)).suffix.lower()
    header_arg = 0 if has_header else None
    notes: list[str] = []

    if suffix in EXCEL_SUFFIXES:
        sheets = pd.read_excel(source, header=header_arg, sheet_name=None)
        first, frame = next(iter(sheets.items()))
        if len(sheets) > 1:
            notes.append(
                f"This workbook has {len(sheets)} sheets ({list(sheets)}). Only "
                f"'{first}' was read."
            )
        result = LoadResult(
            frame, Path(str(name)).name, had_header=has_header, notes=notes
        )
    elif suffix in CSV_SUFFIXES or suffix == "":
        raw = _read_bytes(source)
        text, used_encoding = _decode(raw, encoding)
        if used_encoding == "latin-1" and encoding is None:
            notes.append(
                "Decoded as latin-1, which never fails and therefore proves nothing "
                "about the real encoding. Check for damaged characters."
            )
        used_delimiter = delimiter or sniff_delimiter(text[:8192])

        skipped = skip_rows if skip_rows is not None else sniff_preamble(
            text, used_delimiter
        )
        if skipped:
            notes.append(
                f"Skipped {skipped} line(s) above the header: their field count did "
                "not match the rest of the file, which is what a title or blank row "
                "looks like. Set 'Rows to skip' to 0 to read them as data."
            )

        from io import StringIO

        frame = pd.read_csv(
            StringIO(text),
            sep=used_delimiter,
            header=header_arg,
            skiprows=skipped or None,
            decimal=decimal,
        )
        result = LoadResult(
            frame,
            Path(str(name)).name,
            encoding=used_encoding,
            delimiter=used_delimiter,
            had_header=has_header,
            decimal=decimal,
            skipped_rows=skipped,
            notes=notes,
        )
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Expected CSV/TSV/TXT or Excel."
        )

    if not has_header:
        result.frame.columns = [f"column_{i + 1}" for i in range(result.frame.shape[1])]

    unique, renamed = make_names_unique([str(c) for c in result.frame.columns])
    if renamed:
        result.frame.columns = unique
        notes.append(
            f"Renamed {len(renamed)} duplicate column name(s) so each can be "
            f"addressed: {renamed}. Two columns with one name means one of them is "
            "not the one you think it is."
        )

    if result.frame.empty:
        notes.append("The file parsed to zero rows.")
    if result.frame.shape[1] == 1 and suffix in CSV_SUFFIXES:
        notes.append(
            "Only one column was produced, which usually means the delimiter is "
            "wrong. Try forcing it."
        )
    return result


def make_names_unique(names: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Give every column a distinct name, reporting what had to change.

    Duplicate names are not a cosmetic problem: `frame["Amount"]` returns a
    *DataFrame* rather than a Series when two columns share the name, so every
    downstream operation that expects one column raises "the truth value of a Series
    is ambiguous" -- including the check whose job is to report the duplication.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    # A list, not a dict keyed by the original name: three columns called "Amount"
    # produce two renames, and a dict would keep only the last.
    renamed: list[tuple[str, str]] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
            continue
        seen[name] += 1
        new = f"{name}__{seen[name]}"
        while new in seen:
            seen[name] += 1
            new = f"{name}__{seen[name]}"
        seen[new] = 1
        renamed.append((name, new))
        out.append(new)
    return out, renamed


def _read_bytes(source: str | Path | BinaryIO) -> bytes:
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)  # Streamlit hands back an already-consumed buffer.
        data = source.read()
        return data.encode("utf-8") if isinstance(data, str) else data
    return Path(str(source)).read_bytes()


def _decode(raw: bytes, forced: str | None) -> tuple[str, str]:
    """Decode bytes, returning the text and the encoding that worked."""
    for candidate in ([forced] if forced else list(ENCODINGS)):
        try:
            return raw.decode(candidate), candidate
        except (UnicodeDecodeError, LookupError):
            continue
    raise UnicodeDecodeError(
        "unknown", raw[:16], 0, 1, f"none of {ENCODINGS} could decode this file"
    )
