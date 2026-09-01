# Handoff — what the first attempt taught us

**Read this first.** It is the bridge between the previous project and this one.

- **Previous project:** `~/Documents/masters/diploma` (working name `DataForge`). Still
  there, untouched, on branch `implementing_data_collection`. Use it as a reference
  library, not as a foundation.
- **This project:** a deliberately simpler system that does the same job in a way the
  author can explain line by line.
- **Decision date:** 2026-08-27.

---

## 1. Why we started again

The first attempt was finished — all eight build steps, 45 modules, ~9,300 lines of
library code, 309 passing tests, seven experiments with real measured results. It
works. It is not a failure.

We restarted for one reason: **the author did not write it and cannot defend it.**
A master's defence grades what you can explain. 45 modules of code that was generated
rather than authored is a risk no amount of test coverage removes.

Two secondary reasons, both real:

1. **It drifted from the задание.** The annotation says the practical part is
   *"модулно приложение (пайплайн), което **приема сурови набори от данни**"* — takes
   raw data sets. The first attempt is hardcoded to three specific corpora with
   bespoke loaders per file. It cleans three known things, not whatever you give it.
   The adversarial review in the old repo (`docs/review-01.md`) actually said this:
   its verdict on the file-upload path was *"PROMOTE to primary. Cheapest entry
   point, and what the annotation literally describes."* That was never done.
2. **It went past the mandated stack.** The annotation says
   *"разчитайки на **Pandas и NumPy** за логиката по обработка на масивите от данни."*
   The first attempt leans on scikit-learn, SciPy, Pandera and Pydantic. Not wrong,
   but nobody asked for it, and every extra library is another thing to defend.

## 2. What this project is

A pipeline that takes a CSV or Excel file it has never seen, works out what is wrong
with it, fixes what it can, and shows you the before and after — through a Streamlit
interface, with each step visible and explained.

**Design rules, fixed now:**

1. **Pandas and NumPy do the data work.** Other libraries only where they are not
   doing data processing: Streamlit for the UI, openpyxl to read `.xlsx`, and one
   simple model for the "did it help?" check.
2. **Every method must be explainable in two sentences.** If it cannot be, it is out
   of scope — no matter how good it is.
3. **Every module gets its explanation written at the same time**, in `writing/`.
   Not afterwards. This is how you avoid finishing the code and facing a blank
   70 pages.
4. **Split before fit.** Anything learned from data — a median used to fill blanks, a
   category frequency — is computed from the training half only. Cheap to do, strong
   to defend.
5. **Any file in, no hardcoded datasets.** Sample files live in `project/data/input/`
   as convenience, never as an assumption.

## 3. Scope — the four mandated topics

The задание names four theoretical topics. All four get covered. The theoretical part
("В **теоретичната** част се изследват съвременните методи…") is what must survey the
modern literature — the *implementation* is allowed to be simpler, provided we say so
explicitly in Chapter 4.

| Topic | What we implement | What we survey but do not implement |
|---|---|---|
| Липсващи стойности | disguised-missing detection (`?`, `N/A`, `-999`, `unknown`, blanks); missingness pattern report; group-wise median/mode fill; "was missing" indicator columns | MICE + Rubin's rules, GAIN / MIWAE / VAE imputation |
| Откриване на аномалии | IQR rule, z-score, MAD-based modified z-score; capping (winsorising) rather than deletion | Isolation Forest, LOF, autoencoder methods |
| Премахване на дубликати | exact duplicates; duplicates after normalising (casefold, trim, collapse whitespace); optional near-duplicate text by word-overlap similarity | MinHash + LSH banding, transformer-based entity resolution (Ditto) |
| Инженеринг на признаци | skew measurement + log transform; date decomposition (year/month/weekday/is-weekend); one-hot for low cardinality, frequency encoding for high; scaling; correlation pruning | automated feature synthesis, deep representation learning |

**Note on anomalies.** The first attempt nearly missed this topic entirely, because the
English translation of the annotation in its repo silently dropped *"откриване на
аномалии"*, and the requirements table was built from that translation. Do not repeat
this: the Bulgarian original is the source of truth. It is reproduced in
`writing/00-zadanie.md`.

## 4. What we deliberately do NOT build

Each of these was built in the first attempt. Each is being dropped on purpose, and
each drop is thesis material — "we considered X and chose Y because Z" is a good
answer at a defence.

| Dropped | Why |
|---|---|
| MICE + Rubin's pooling (680 lines) | Statistically correct and genuinely better, but not defensible by someone who did not derive it. Chapter 2 explains what it does and why it beats single imputation; we implement the simpler method and say so. |
| MinHash + LSH (~1,000 lines) | Same reasoning. The maths (banding, the S-curve, `h = b·r`) is Chapter 2 material; word-overlap similarity is what we run. |
| Little's MCAR test (398 lines) | Easy to get subtly wrong — a broken implementation prints a plausible p-value. Also degenerate on the census corpus, whose missingness is entirely in categorical columns. |
| Canonical content hashing + determinism harness (~420 lines) | Nobody asked for it. It works well and proved itself, but it answers a question the задание never poses. |
| Operation ABC + plugin registry + Pydantic recipe schema + fitted-state store (~1,500 lines) | Architecture for a system with many pluggable operations. We have about a dozen fixed steps in a fixed order. A plain, readable pipeline is the right shape. |
| Pandera data contracts (332 lines) | Was also the least-tested part of the old project (0% coverage). A simple column-rule check in pandas is enough. |
| Operation-level provenance / Mermaid lineage graphs | Nice to look at, adds nothing to the four mandated topics. |
| The compensatory quality score | It was **broken**: penalties summed per dimension with a 0.30 floor per critical finding, so it reported *completeness 10.0%* on a corpus that is 99.12% populated, and it was non-monotone in the data. If we report quality numbers, they are plain measured shares. |

## 5. What we carry over

Finished work worth reusing, all in the old repo:

| From | What it is |
|---|---|
| `docs/glossary.md` | ~40 EN→BG terms, already decided. There is no settled Bulgarian for *near-duplicate*, *data contract*, *fitted state* — deciding once prevents visible inconsistency across 75 pages, and it becomes an appendix. Copied to `writing/glossary.md`. |
| `docs/review-01.md` | An adversarial review of the original design. Excellent Chapter 2 ammunition and a source of honest limitations. |
| `docs/figures.md` | The frozen 15-figure list, with the warning that each finished figure costs ~0.25 days. |
| `docs/limitations.md` | The *pattern* — limitations plus the defence question each invites. Worth imitating, not copying. |
| `dataforge/ops/basic.py` | The disguised-missing token list. |
| `dataforge/profiling/detectors/builtin.py` | The logic of nine detectors, to reimplement simply. |
| `data/raw/adult/` | The census dataset — 48,842 rows, and it stores unanswered questions as a literal `?`, so a naive load reports **zero** missing values on a file where 3,620 rows have missing answers. The best demo file we have. |

## 6. Things learned the hard way — do not rediscover these

- **Homebrew's `python@3.11` cannot create a virtualenv on this machine.** Its
  `pyexpat` wants a newer `libexpat` than macOS ships. Use
  `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.
- **That Python has no root certificates**, so any download fails with
  `CERTIFICATE_VERIFY_FAILED` until you
  `export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")`.
- **Streamlit is interactive and runs real Python on every click.** You do not need
  FastAPI. The задание mandates Streamlit anyway.
- **Declare every dependency you import.** The old project imported `yaml` and `scipy`
  without declaring them; a clean install could not even import the library.
- **Commit and push from day one.** A week of work in the old project existed only
  inside a remote container and was nearly lost — nothing had been pushed.
- **One piece of evidence beats none.** Train a simple model on raw vs cleaned data
  and report both scores. Without it you can only say the data *changed*, not that it
  *improved*. Someone will ask.

### 6b. Learned while building *this* version (steps 1–10)

Added 2026-09-01. These are the mistakes this project actually made, and the checks that
caught them. They are also defence material: "here is a bug I found in my own work and
how" answers a question no amount of clean code does.

**A test that cannot fail is worse than no test.** This happened three times.

- The fixture for "both evaluation arms use the same held-out rows" had no duplicate
  rows, so the deduplication step never fired and the test passed with the bug
  reintroduced.
- The check for "don't celebrate a gain between two sub-chance models" only asserted
  when a sample file happened to score below 0.5 — which none do.
- The claim in Р13 that a test guards the dependency boundary was not true until the
  test was written.

**The habit that fixes it: put the bug back and watch the test fail.** If it passes, the
test is decoration. Do this every time a test is written *for* a specific bug.

**Move the judgement out of the view so it can be tested.** The "is this an
improvement?" decision lived in `app.py`, where only a lucky sample file could exercise
it. Moved to `evaluate.verdict()`, all six branches became directly testable — and the
exported report, which had been wording its own verdict from the sign of the difference,
stopped contradicting the page.

**Two outputs of one tool must read the same source.** The app said "worse than a coin
toss"; the report called the same number an improvement. Both now call `verdict()`.

**Silent index renumbering breaks anything that refers to rows later.** Four cleaning
steps called `reset_index(drop=True)`. Nothing depended on it until the evaluation needed
to know which rows were held back — and because the labels still looked plausible
(unique, ascending, in range) the two arms were scored on *different* test sets while
producing believable numbers. The same root cause silently defeated the totals-row
removal: labels read as positions, out of range after an earlier row was dropped, "no
matching rows" reported, and the totals row survived into the cleaned data.

**Inherited documents drift into lies.** `writing/figures.md` was marked *frozen* while
describing the first project's LSH curves, `Operation` registry and MICE experiments.
A stale document that looks authoritative is worse than a missing one. Re-read every
inherited file against the current code before using it — the same class of error as Р8,
where a requirement was taken from a summary instead of the задание.

**Hand-drawn architecture diagrams are wrong almost immediately.** The module graph had
*every arrow reversed*: a data-flow picture and an import graph look identical until
compared with the source. It is now generated from the imports.

**Verify by doing, not by asserting.** `RUNNING.md` was checked by making a clean copy of
the tree, a fresh venv, and following it literally. That surfaced something no
development environment could: pip resolves **pandas 3.0.5 / numpy 2.5.2** for a new
reader, and all tests still pass. Likewise, every figure caption number was checked
against `measurements.csv` — two were wrong, copied from a trial run.

**Streamlit specifics worth not rediscovering.**
- HTTP 200 proves nothing; exceptions render *inside* the page. Use `AppTest`.
- `AppTest` is not sufficient either — it simulates a rerun on every interaction, which
  a real `st.form` never does, so it cannot see form-batching bugs. Check the browser.
- One name assigned both at module level and inside a `with tab:` block silently changes
  meaning for every later tab. There is a test for this.
- A button reruns the script and `st.tabs` loses the active tab, so the user is thrown
  back to the first tab and never sees the result. Cache instead of using a button.

**Write the expected result down before measuring it.** The pre-registration in
`writing/05-results/01-evidence.md` was committed before the evaluation module existed.
One of its three predictions was then **refuted** — and because it was already committed,
recording the contradiction was the only honest option. That refuted prediction is now
the most interesting result in Chapter 5. See Р12.

**When the browser tooling disconnects**, `list_connected_browsers` then
`select_browser` recovers it; a locked laptop drops the extension.

## 7. The honest finding from the old project's experiments

Worth knowing, because it should shape what this project claims.

The old project ran a proper experiment: fixed held-out test set, five arms, two
models, paired comparisons with multiplicity correction. Result: **on clean data,
cleaning made no measurable difference** — every arm within 0.002 AUC, and an arm that
simply tuned the model on raw data did marginally *better*.

But when the data was progressively corrupted, the pipeline's advantage grew
monotonically: +0.0029 at 10% corruption, +0.0041 at 20%, +0.0048 at 40%. And the
naive "just drop rows with blanks" approach collapsed from 27,082 usable training rows
to **21** at 40% corruption.

**So the defensible claim is conditional: data preparation pays in proportion to how
much is wrong with the data.** Not "cleaning always helps". State it that way and it
is both true and interesting. Claim more and it is refutable with your own numbers.

## 8. Structure

```
datacleaner/
├── HANDOFF.md          this file
├── PLAN.md             the 11 steps, anti-drift rules, and what each step produced
├── RUNNING.md          how to run it — verified from a clean tree
├── BACKLOG.md          ideas deliberately not built (rule 5)
├── scripts/figures.py  regenerates every plot and the module graph
├── project/            the code
│   ├── src/            the pipeline
│   ├── data/input/     drop CSV / XLSX here
│   ├── data/output/    cleaned results land here
│   └── tests/
└── writing/            the wording, in Bulgarian, for the thesis
    ├── 00-zadanie.md   the annotation, Bulgarian original = source of truth
    ├── 02-theory/      one file per mandated topic
    ├── 03-design/      requirements, architecture
    ├── 04-implementation/  one explanation per module
    ├── 05-results/     before/after numbers, and the pre-registration
    ├── figures/        the ten figures, Bulgarian captions, measurements.csv
    ├── decisions.md    running log of "we chose A over B because…" (Р1–Р13)
    └── glossary.md     EN→BG terminology
```
