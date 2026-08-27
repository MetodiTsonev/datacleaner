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


def read_table(
    source: str | Path | BinaryIO,
    *,
    has_header: bool = True,
    encoding: str | None = None,
    delimiter: str | None = None,
) -> LoadResult:
    """Read a CSV or Excel file into a frame, recording how it was read.

    Args:
        source: a path, or an open binary file object (Streamlit's uploader).
        has_header: whether the first row holds column names. When False,
            columns are named `column_1 ... column_n`; a wrong guess here is
            silent and destructive, so it is never inferred.
        encoding: force an encoding instead of trying `ENCODINGS`.
        delimiter: force a CSV delimiter instead of sniffing.
    """
    name = getattr(source, "name", None) or str(source)
    suffix = Path(str(name)).suffix.lower()
    header_arg = 0 if has_header else None
    notes: list[str] = []

    if suffix in EXCEL_SUFFIXES:
        frame = pd.read_excel(source, header=header_arg)
        result = LoadResult(frame, Path(str(name)).name, had_header=has_header, notes=notes)
    elif suffix in CSV_SUFFIXES or suffix == "":
        raw = _read_bytes(source)
        text, used_encoding = _decode(raw, encoding)
        if used_encoding == "latin-1" and encoding is None:
            notes.append(
                "Decoded as latin-1, which never fails and therefore proves nothing "
                "about the real encoding. Check for damaged characters."
            )
        used_delimiter = delimiter or sniff_delimiter(text[:8192])
        from io import StringIO

        frame = pd.read_csv(StringIO(text), sep=used_delimiter, header=header_arg)
        result = LoadResult(
            frame,
            Path(str(name)).name,
            encoding=used_encoding,
            delimiter=used_delimiter,
            had_header=has_header,
            notes=notes,
        )
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Expected CSV/TSV/TXT or Excel."
        )

    if not has_header:
        result.frame.columns = [f"column_{i + 1}" for i in range(result.frame.shape[1])]

    if result.frame.empty:
        notes.append("The file parsed to zero rows.")
    if result.frame.shape[1] == 1 and suffix in CSV_SUFFIXES:
        notes.append(
            "Only one column was produced, which usually means the delimiter is "
            "wrong. Try forcing it."
        )
    return result


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
