"""Build the deliberately dirty demo file.

Written once, committed as data, and kept as a script so the file's contents are
documented rather than mysterious. Every defect below is planted on purpose and
matches one check in src/detect.py, so the demo exercises the whole detector set
-- the census file alone triggers only half of them.

Also carries a date column, which the census file lacks, so the datetime features
in stage 7 have something to work on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260827)
N = 240

cities_clean = ["Sofia", "Plovdiv", "Varna", "Burgas", "Ruse"]
# Planted: the same city spelled several ways -> inconsistent_categories
cities_dirty = {
    "Sofia": ["Sofia", "sofia", " Sofia", "SOFIA", "Sofia "],
    "Plovdiv": ["Plovdiv", "plovdiv", "Plovdiv"],
    "Varna": ["Varna", "varna "],
    "Burgas": ["Burgas"],
    "Ruse": ["Ruse", "ruse"],
}

base_city = rng.choice(cities_clean, N, p=[0.4, 0.2, 0.2, 0.1, 0.1])
city = [rng.choice(cities_dirty[c]) for c in base_city]

frame = pd.DataFrame(
    {
        # Planted: unique per row -> identifier
        "order_id": [f"ORD-{i:05d}" for i in range(N)],
        # Planted: real dates, for the stage-7 datetime features
        "order_date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 540, N), unit="D"),
        "city": city,
        # Planted: strongly skewed, all non-negative -> high_skew, log applicable
        "amount": np.round(rng.lognormal(3.2, 1.1, N), 2),
        # Planted: a genuine continuous column with a few extreme values -> outliers
        "delivery_days": np.clip(rng.normal(4, 1.5, N), 0.5, None).round(1),
        # Planted: short discrete scale -> outlier rules must decline (info only)
        "rating": rng.integers(1, 6, N),
        # Planted: one value throughout -> constant_columns
        "currency": ["BGN"] * N,
        # Planted: no value at all -> empty_columns
        "notes": [None] * N,
        # Planted: numbers and text in one column -> mixed_types
        "weight_kg": [
            str(round(float(w), 1)) if rng.random() > 0.25 else
            rng.choice(["heavy", "light", "n/a", "approx 5"])
            for w in rng.normal(8, 3, N)
        ],
        # Planted: boolean written as text
        "paid": rng.choice(["yes", "no"], N, p=[0.8, 0.2]),
        # Planted: numbers written for people -> numeric_in_text (comma decimal and
        # a thousands space, the European convention including Bulgarian)
        "invoice_total": [
            f"{v:,.2f}".replace(",", " ").replace(".", ",")
            for v in rng.lognormal(6.5, 0.8, N)
        ],
        # Planted: text encoded as UTF-8 then read as cp1252 -> encoding_damage
        "supplier": rng.choice(
            ["Ð¡Ð¾Ñ„Ð¸Ñ ÐžÐžÐ”", "Ð’Ð°Ñ€Ð½Ð° ÐÐ”", "Ð ÑƒÑ Ðµ ÐžÐžÐ”"], N
        ),
        # Planted: two date layouts in one column -> mixed_date_formats
        "paid_date": [
            (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(d))).strftime(
                "%Y-%m-%d" if i % 2 else "%d.%m.%Y"
            )
            for i, d in enumerate(rng.integers(0, 500, N))
        ],
        # Planted: mostly real names, a few one-character stubs -> short_values
        "contact": [
            rng.choice(["Ivan Petrov", "Maria Ivanova", "Georgi Dimitrov"])
            if rng.random() > 0.08
            else rng.choice(["x", "-", "?"])
            for _ in range(N)
        ],
        # Planted: an address with an embedded newline -> control_characters
        "address": [
            "ul. Vitosha 1\nSofia" if rng.random() < 0.06 else "ul. Vitosha 1, Sofia"
            for _ in range(N)
        ],
        # The target for stage 8
        "returned": rng.choice(["no", "yes"], N, p=[0.75, 0.25]),
    }
)


# Planted: disguised missing values, three different tokens in three columns
frame.loc[rng.choice(N, 18, replace=False), "city"] = "?"
frame.loc[rng.choice(N, 12, replace=False), "paid"] = "N/A"
amount_missing = rng.choice(N, 15, replace=False)
frame["amount"] = frame["amount"].astype(object)
frame.loc[amount_missing, "amount"] = "-999"

# Planted: genuine nulls as well, so both missing-value checks fire
frame.loc[rng.choice(N, 20, replace=False), "delivery_days"] = np.nan

# Planted: extreme but real values -> the IQR rule should catch these
frame.loc[[3, 77, 150], "delivery_days"] = [41.0, 38.5, 44.2]

# Planted: 14 exact duplicate rows, plus 6 that differ only by case/spacing
frame = pd.concat([frame, frame.iloc[:14]], ignore_index=True)
extra = frame.iloc[20:26].copy()
extra["city"] = extra["city"].astype(str).str.upper() + " "
frame = pd.concat([frame, extra], ignore_index=True)

# Planted: a completely empty row -> empty_rows
frame = pd.concat(
    [frame, pd.DataFrame([{c: None for c in frame.columns}])], ignore_index=True
)

# Planted: a totals row at the foot -> summary_rows
total = {c: None for c in frame.columns}
total["city"] = "TOTAL"  # pre-rename name
total["amount"] = 999999.0
total["delivery_days"] = float(pd.to_numeric(frame["delivery_days"], errors="coerce").sum())
frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)

# Planted: messy header names -> column_names (trailing space, embedded newline).
# Done last: every assignment above addresses columns by name, and renaming first
# made `frame.loc[:, "city"] = "?"` create a new column instead of modifying this one.
frame = frame.rename(columns={"city": "city ", "rating": "rating\n"})

out = Path(__file__).parent / "project" / "data" / "input" / "messy-orders.csv"

# Planted: a title row and a blank row above the header -> the loader's preamble
# detection. Written by hand because pandas cannot emit rows above its own header.
body = frame.to_csv(index=False)
out.write_text("Orders export 2024 - internal use only\n\n" + body, encoding="utf-8")
print(f"wrote {out}  {frame.shape[0]} rows x {frame.shape[1]} columns (+2 preamble)")
