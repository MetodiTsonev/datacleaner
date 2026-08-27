# DataCleaner — implementation plan

## Context

The first attempt (`~/Documents/masters/diploma`) is finished and works — 45 modules,
~9,300 lines, 309 tests, seven measured experiments. It is being set aside because the
author did not write it and cannot defend it line by line, and because it drifted from
the задание in two concrete ways: it is hardcoded to three specific corpora rather than
accepting arbitrary raw files, and it leans on scikit-learn/SciPy/Pandera/Pydantic when
the задание names **Pandas and NumPy**.

This plan builds the replacement at `~/Documents/masters/datacleaner` (already
scaffolded and committed as `a3a2a07`). Target: a system the author can explain in full,
covering all four mandated topics, at roughly **2,300 lines including tests** instead of
16,000.

**Decisions already taken** (see `writing/decisions.md` Р1–Р7, and answers this session):
Streamlit UI in **English**; the evidence check written **from scratch in NumPy** so the
whole project is literally only Pandas + NumPy + Streamlit; **drop-folder input only**,
no live collector.

**The intended outcome is not just working code.** It is working code plus a Bulgarian
explanation of every part, written as we go, so the thesis text is assembled rather than
faced from scratch. The first attempt reached 100% code and 0 pages.

---

## ⚠ Correction, 2026-08-27 — a mandated verb was missing

This plan originally had eight stages and **no validation stage**. The задание says the
system automates *"почистване, трансформиране и **валидиране** на данни"* — three verbs,
and the third had no owner.

The checks in stage 3 are automatic **discovery**: the system deciding for itself what
looks wrong. Validation is a different thing — checking data against rules **someone
declared**: `age > 0`, `city ∈ {…}`, `order_id` unique, `end_date >= start_date`.
Conflating them hid the fact that one of three mandated verbs was not implemented.

Same class of error as the first project's omission of *откриване на аномалии*: a
requirement with no module, no chapter section, no experiment. Found by the author
asking why the checks did not cover ordinary constraints.

**Correction:** stage 4 is now validation; the pipeline has nine stages. Recorded in
`writing/decisions.md` Р8 so the correction is traceable rather than silently absorbed.

Two scope additions agreed at the same time (Tier 1):

- **Six format-hygiene checks.** Measured gaps, none of them caught: leading/trailing
  whitespace; single characters in a longer text column; `1 234,56` / `$100` / `12 lv`
  (numeric quantities arriving as text — the worst, since such a column is then one-hot
  encoded instead of averaged); mojibake from double-decoded UTF-8; mixed date layouts
  in one column; control characters.
- **Ingestion and correctness fixes.** The target column was unprotected — stage 6
  would have median-imputed it, fabricating labels. Plus junk rows at the top and
  bottom of exports (title rows, blank rows, `TOTAL` rows), messy and duplicate column
  names, the decimal separator, and fully empty rows.

---

## Anti-drift rules — the reason this plan exists

These are binding. The previous project ballooned because none of them existed.

1. **One step at a time, and a step is not finished until its `writing/` file exists.**
   This is the hard gate. No starting step N+1 with step N's explanation unwritten.
2. **Line budgets per module** are given below. Exceeding a budget by more than ~50% is a
   signal to stop and reconsider, not to keep going.

   ① `checks.py` is the one accepted overrun, agreed explicitly on 2026-08-27 rather
   than quietly revised. It holds 21 independent check functions; ~40% of its lines are
   the user-facing messages that explain each finding, which are the product rather
   than overhead, and the logic is ~24 lines per check. Two checks were merged before
   accepting it (`empty_columns` + `constant_columns`). No other budget may be raised
   without the same explicit agreement — a budget revised whenever it is breached is
   not a budget.
3. **Every component must trace to a requirement in `writing/00-zadanie.md`.** If it
   doesn't, it isn't built. No exceptions for interesting ideas.
4. **The dependency list is closed:** `pandas`, `numpy`, `streamlit`, `openpyxl`,
   `pytest`. Adding anything requires a `writing/decisions.md` entry first.
5. **Ideas during implementation go to `BACKLOG.md`, never straight into the code.**
6. **No "while I'm here" refactors.** Note it, move on.
7. **Explicitly not building:** multiple imputation / Rubin's rules, MinHash/LSH,
   Little's MCAR test, content hashing or determinism harness, plugin registry or
   Operation ABC, Pydantic schemas, data contracts, lineage graphs, composite quality
   scores, a recommender with topological sorting. All are surveyed in Chapter 2 as
   theory and named as not-implemented in Chapter 4. See `HANDOFF.md` §4.

---

## Target structure

```
project/
├── app.py                Streamlit UI                        ~350
├── pyproject.toml        minimal, for `pip install -e .`
├── src/
│   ├── loader.py         read CSV/XLSX, encoding fallback     ~90
│   ├── profile.py        semantic type inference + stats     ~170
│   ├── detect.py         the Finding contract + the runner     ~90
│   ├── checks.py         the checks, grouped by topic         ~1000 ①
│   ├── validate.py       declared constraints + quarantine    ~200
│   ├── plan.py           Findings -> ordered repair steps    ~130
│   ├── clean.py          structure, duplicates, missing      ~320
│   ├── anomalies.py      IQR / z-score / MAD, capping        ~150
│   ├── features.py       skew, dates, encoding, scaling      ~240
│   ├── evaluate.py       NumPy logistic regression + AUC     ~160
│   └── report.py         before/after tables, export         ~140
└── tests/                                                    ~450
```

Reuse from the old project — read these, reimplement simply, do **not** copy wholesale:

- `diploma/dataforge/profiling/detectors/base.py` — the `Finding` + `Suggestion`
  dataclass pattern. Worth keeping: a finding carries a *suggested repair*, which is what
  makes the system prescriptive rather than merely descriptive. Drop the ABC, the
  registry, and the ISO dimension vector.
- `diploma/dataforge/ops/basic.py` — `DEFAULT_TOKENS`, the disguised-missing token list.
- `diploma/dataforge/profiling/detectors/builtin.py` — the logic of the nine detectors,
  including the `hours_per_week` self-disclaiming outlier case worth preserving.
- `diploma/dataforge/io.py` — the encoding-fallback idea only (BBC data is cp1252 and
  breaks a naive `read_csv`).

---

## Step 0 — Environment and skeleton

- venv from `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (Homebrew
  3.11 is broken on this machine — see `HANDOFF.md` §6).
- `requirements.txt` + minimal `pyproject.toml`; `pip install -e .`.
- `app.py`: upload or pick a file from `data/input/`, show `df.head()`.

**Done when:** `streamlit run app.py`, drop in `data/input/adult-census.csv`, see rows.
**Writing:** `writing/03-design/01-architecture.md` — the module map, the eight pipeline
stages, and why this shape rather than a plugin architecture.

## Step 1 — Loader and profiler

- `loader.read_table()`: CSV with encoding fallback (utf-8 → cp1252 → latin-1, recording
  which worked), XLSX via openpyxl. Returns frame + metadata.
- `profile.py`: infer a **semantic type** per column — numeric / categorical / datetime /
  boolean / text / constant — by attempting conversion and thresholding the success rate.
  Per column: dtype, missing count and share, distinct count, top values, and for
  numerics min/max/mean/median/std/skew/zero-share.

This is where "accepts arbitrary raw files" actually lives, so type inference must be
conservative and explainable. State the thresholds and why.

**Done when:** any CSV/XLSX produces a correct column-by-column profile; the census file
resolves as 6 numeric + 9 categorical.
**Writing:** `writing/04-implementation/01-loader-profile.md`.

## Step 2 — Detectors

Eight checks, each returning `Finding(name, severity, columns, message, affected_rows,
affected_share, evidence, suggestion)`:

1. **disguised missing** — `?`, `N/A`, `-999`, `unknown`, blanks masquerading as data
2. **missing values** — real nulls, per column
3. **exact duplicates** — identical rows
4. **normalised duplicates** — identical after casefold / trim / collapse whitespace
5. **outliers** — IQR rule and MAD-based modified z-score
6. **constant / quasi-constant columns**
7. **inconsistent categories** — same category, different spelling or case
8. **mixed types** — numeric-looking strings mixed with text in one column

**Done when:** the census file reproduces its known pathologies — disguised missing in
`workclass` 2,799 / `occupation` 2,809 / `native_country` 857 while pandas reports zero
nulls; 52 exact duplicate rows; `education` ↔ `education_num` redundancy;
`capital_gain` skew 11.89.
**Writing:** `writing/04-implementation/02-detect.md`, plus start
`writing/02-theory/02-anomalies.md` (the MAD modified z-score is theory, not just code).

## Step 3 — Validation against declared constraints

The mandated verb. Small and declarative.

- **Rule types:** not-null, unique, numeric range, allowed value set, regex pattern,
  expected type, and one cross-column comparison (`a <= b`).
- **Inferred as a draft** from the profile, then **editable by the user** in the UI.
  Inference cannot tell an intentional constraint from an accident of this batch, and
  must say so.
- **Output is a split, not a verdict:** valid rows, and rejected rows each carrying the
  reason they failed. Rejected rows export separately — the quarantine file.

**Done when:** a rule added in the UI (`amount > 0`) immediately reports which rows
fail and why, and those rows can be exported. This is the ten-second answer to *"could
you add a constraint?"* at the defence.
**Writing:** `writing/04-implementation/03-validate.md`,
`writing/02-theory/05-validation.md`.

## Step 4 — Repair plan

- A **fixed, sensible order** of stages, filtered to the ones the findings require. No
  topological sort — the order is a documented design decision, not a computation.
- Each step carries a plain-English rationale shown in the UI.
- **Merging:** three disguised-missing findings become one operation.
- **Derivation:** replacing `?` with real nulls *creates* a new problem, so an imputation
  step becomes necessary that no detector asked for. Keep this — it is the clearest
  demonstration that the system reasons rather than pattern-matches, and it was a
  genuine finding from the first attempt.
- Each step is marked pre-split or post-split.

**Done when:** the census file produces an ordered plan containing the derived
imputation step.
**Writing:** `writing/04-implementation/03-plan.md`.

## Step 5 — Cleaning: structure, duplicates, missing values

Pre-split (nothing learned from data): drop constant columns → normalise categories →
replace disguised missing → drop duplicates.

**Then the train/test split** (pandas/numpy, stratified when a target is named).

Post-split, fitted on the training half only: impute — global median/mode, optional
**group-wise median** where a sensible grouping column exists, plus `was_missing`
indicator columns.

**The target column is protected throughout.** Never imputed, never transformed, never
scaled, never encoded as a feature. Rows whose target is missing are **dropped**,
because a guessed label is worse than a missing row. Nothing in the original plan
prevented median-imputing the target, which would have fabricated labels.

**Done when:** census cleaned to zero nulls with the 52 duplicates gone, **and a test
proves the fill value equals the training median, not the full-data median.** That test
is the leakage guarantee and it is worth one sentence at the defence.
**Writing:** `writing/04-implementation/04-clean.md` and
`writing/02-theory/01-missing-values.md` — the latter must survey MICE, GAIN/MIWAE/VAE
and say plainly that we implement the simpler method and why.

## Step 6 — Anomalies

- IQR bounds, z-score, MAD-based modified z-score; state why MAD is more robust on
  skewed data.
- **Cap (winsorise) rather than delete**, and report how many values were capped per
  column.
- Carry over the honest case: on `hours_per_week` a single value covers 47% of the
  column, so the IQR rule flags 13.6% of rows and the count must not be read as an error
  count. A detector that disclaims its own rule is a strength — keep it.

**Done when:** capping applied to `capital_gain` and `fnlwgt` with counts reported, and
`hours_per_week` flagged as unreliable rather than silently "fixed".
**Writing:** finish `writing/02-theory/02-anomalies.md`, add
`writing/04-implementation/05-anomalies.md`.

## Step 7 — Feature engineering

- Measure skew, apply `log1p` where high, **report skew before and after** —
  `capital_gain` 11.89 → ~0 is figure material.
- Datetime decomposition: year, month, weekday, is-weekend.
- Categoricals: one-hot when few distinct values, frequency encoding when many.
- Standard scaling in NumPy; correlation pruning.

**Gap to close in this step:** the census file has **no date column**, so datetime
features cannot be demonstrated on it. Build a second small, deliberately dirty demo
file (~50 rows) with dates, mixed category spellings and disguised blanks. It doubles as
a clean test fixture and as a thesis figure.

**Done when:** census becomes an all-numeric matrix with no text columns, skew reduced,
feature count reported before and after.
**Writing:** `writing/02-theory/04-features.md`,
`writing/04-implementation/06-features.md`.

## Step 8 — Evidence that cleaning helped

Pure NumPy, ~160 lines: stratified split, logistic regression by gradient descent on
standardised inputs, ROC AUC. Compare raw (minimally encoded so a model can run at all)
against cleaned.

**Frame the result honestly before seeing it.** The first attempt measured this properly
and found that on *clean* data preparation made no measurable difference, while its
advantage grew monotonically as data was corrupted (+0.0029 at 10%, +0.0041 at 20%,
+0.0048 at 40%), and that naive row-dropping collapsed from 27,082 usable rows to 21.
So the claim is **conditional**: preparation pays in proportion to how much is wrong with
the data. Write that down before running anything, so the conclusion isn't retrofitted.

A cheap, strong addition: corrupt the demo file at 0/10/20/40% and show the same trend.
That converts the honest caveat into your own measured finding.

**Done when:** two AUC numbers appear in the app, and a test verifies the NumPy AUC
against a hand-computed small case.
**Writing:** `writing/04-implementation/07-evaluate.md`,
`writing/05-results/01-evidence.md`.

## Step 9 — The Streamlit app

Tabs, left to right, mirroring the pipeline: **Data → Profile → Findings → Plan → Run →
Before/After → Evidence → Export**.

- Interface text in **English**; column names stay as they appear in the uploaded file.
- The Run tab shows each step as it completes, with its rationale — this is the
  "see each step" requirement.
- Export the cleaned file plus a summary of what was done.

**Done when:** the whole flow works in a browser from upload to export without touching
the terminal.
**Writing:** `writing/04-implementation/08-app.md`.

## Step 10 — Consolidate

- Tests green across the core modules.
- Generate the figures from `writing/figures.md` this project actually supports. Each
  gets a **Bulgarian caption** explaining the English screen — that is how an English UI
  sits correctly inside a Bulgarian thesis.
- `README.md` and a short `RUNNING.md` verified by following them from scratch.

**Done when:** `pytest` passes, the figures exist, and the README works for a stranger.

## Step 11 — Thesis text

Assemble chapters from `writing/`. Planned separately — it is the largest remaining
piece of work and it is not code.

**Blocking and still unknown:** the faculty formatting template, and whether the full
official задание says more than the annotation paragraph. Both must be obtained before
prose starts; retrofitting a template into finished pages costs days.

---

## Verification

Per step, the "Done when" above is the test. End to end, when all steps are complete:

1. `cd ~/Documents/masters/datacleaner/project && streamlit run app.py`
2. Upload `data/input/adult-census.csv`. Confirm the Profile tab reports **zero** missing
   values (the disguised-`?` trap) while the Findings tab reports three columns with
   thousands of disguised blanks. That contrast is the single best demonstration in the
   project.
3. Confirm the Plan tab lists steps in order, including the *derived* imputation step.
4. Run it. Confirm zero nulls afterwards, 52 duplicates removed, capping counts shown.
5. Check the Evidence tab shows two AUC numbers.
6. Export, and reopen the exported file to confirm it loads.
7. Upload the small dirty demo file and confirm the date features appear — proving the
   system is not tuned to one dataset.
8. `pytest` from `project/`.
9. Confirm every `src/*.py` has a matching file in `writing/04-implementation/`. If any
   is missing, the anti-drift gate was breached.
