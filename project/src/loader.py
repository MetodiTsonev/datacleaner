"""Reading the input file.

Everything here exists because the app has to take files it has never seen. What's
detected is always reported, never applied silently: encoding, delimiter, whether
there's a header, and how many junk lines sat above it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

# latin-1 last on purpose: it decodes any byte sequence, so it never fails and its
# success proves nothing. We report when it was used.
ENCODINGS = ("utf-8", "cp1252", "latin-1")

#: Candidate CSV delimiters, most likely first.
DELIMITERS = (",", ";", "\t", "|")

PREAMBLE_SCAN_LINES = 20
PREAMBLE_CONFIRM_ROWS = 3  # so a genuine one-column file isn't read as a preamble

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
    """Whichever candidate splits the first line into the most fields.

    csv.Sniffer raises on ambiguous input; we always need an answer.
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
    """Junk lines above the real header - a title row, a blank row.

    The header is the first line whose field count matches what the lines below
    agree on. If line one already matches there's no preamble, which is the normal
    case and has to stay free of false positives.
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


def sniff_has_header(text: str, delimiter: str, *, skip: int = 0) -> bool:
    """Whether row one is column names or data.

    The signal is a type mismatch: a header cell is text, a data cell in a numeric
    column is a number. So if any first-row cell is numeric and its column is mostly
    numeric, that row is data.

    Biased toward True. Reading a header as data destroys a row and still looks like
    a valid table; the other way round only costs readable names.
    """
    from io import StringIO

    reader = csv.reader(StringIO(text), delimiter=delimiter)
    rows = []
    for index, row in enumerate(reader):
        if index < skip:
            continue
        rows.append(row)
        if len(rows) > PREAMBLE_CONFIRM_ROWS + 1:
            break
    if len(rows) < 3:
        return True

    first, body = rows[0], rows[1:]
    width = min(len(first), min(len(r) for r in body))
    for column in range(width):
        head = first[column].strip()
        if not head or not _is_number(head):
            continue
        below = [r[column].strip() for r in body if column < len(r)]
        numeric_below = sum(1 for v in below if v and _is_number(v))
        if below and numeric_below / len(below) >= 0.5:
            return False
    return True


def _is_number(value: str) -> bool:
    try:
        float(value.replace(" ", "").replace(",", "."))
    except ValueError:
        return False
    return True


def probe_header(source: str | Path | BinaryIO, *, sample_bytes: int = 65_536) -> bool:
    """sniff_has_header on a cheap prefix, so the UI can set its default before the
    full read. Leaves an uploaded buffer rewound. True on Excel or on failure."""
    name = str(getattr(source, "name", None) or source)
    if Path(name).suffix.lower() in EXCEL_SUFFIXES:
        return True
    try:
        if hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            raw = source.read(sample_bytes)
            if hasattr(source, "seek"):
                source.seek(0)
            raw = raw.encode("utf-8") if isinstance(raw, str) else raw
        else:
            # `source`, not `name`: for a Path, `.name` is only the final component,
            # so opening it looked in the working directory instead. The first version
            # of this function did exactly that, and the broad `except` below turned
            # the resulting FileNotFoundError into a confident wrong answer.
            with Path(source).open("rb") as handle:  # type: ignore[arg-type]
                raw = handle.read(sample_bytes)
        text, _ = _decode(raw, None)
        delimiter = sniff_delimiter(text[:8192])
        return sniff_has_header(text, delimiter, skip=sniff_preamble(text, delimiter))
    except (OSError, UnicodeDecodeError, csv.Error):
        # Genuinely cannot tell. Assume a header, which is the recoverable direction.
        return True


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

        if has_header and not sniff_has_header(text, used_delimiter, skip=skipped):
            notes.append(
                "The first row looks like data, not column names: some of its values "
                "are numbers in columns that are otherwise numeric. If the column "
                "names below are actually values, untick 'First row contains column "
                "names'."
            )

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

    # Always assign, not just when something was renamed: pandas gives integer labels
    # for numeric headers (a sheet with 2020, 2021, 2022 as columns), and profiles
    # carry names as strings, so every later `frame[name]` lookup would raise KeyError.
    unique, renamed = make_names_unique([str(c) for c in result.frame.columns])
    result.frame.columns = unique
    if renamed:
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
    """Distinct names, plus what changed.

    Not cosmetic: with two columns called Amount, frame["Amount"] returns a DataFrame
    and everything downstream dies on "truth value of a Series is ambiguous" -
    including the check meant to report it.
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
    """Text plus the encoding that worked."""
    for candidate in ([forced] if forced else list(ENCODINGS)):
        try:
            return raw.decode(candidate), candidate
        except (UnicodeDecodeError, LookupError):
            continue
    raise UnicodeDecodeError(
        "unknown", raw[:16], 0, 1, f"none of {ENCODINGS} could decode this file"
    )
