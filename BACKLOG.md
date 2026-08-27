# Backlog

Ideas that came up during implementation and were **not** built, per anti-drift
rule 5 in `PLAN.md`. Nothing here enters the code without first becoming a
requirement traceable to `writing/00-zadanie.md`.

| Idea | Where it came from | Why not now |
|---|---|---|
| `class_imbalance` check | the first attempt had one | It maps to "use a stratified split", but stage 4 stratifies whenever a target is named, so the finding would be informational only. Five lines, no repair to drive. |
| Near-duplicate text detection by word overlap | `HANDOFF.md` §3 lists it as optional | Only worth building if a demo file has a long-text column. The census file has none. Revisit if a text corpus is added. |
| Correlation-based redundancy for numeric pairs | natural extension of `redundant_columns` | The exact 1:1 mapping check covers the census case (`education` ↔ `education_num`). A correlation threshold needs a defensible cut-off, which is a decision without a driving requirement yet. |
| Per-column choice of imputation strategy in the UI | obvious once the plan exists | The задание asks for *automated* procedures. Letting the user pick each strategy turns the system into a toolbox. Possibly a read-only override later. |
| Profiling re-run after cleaning | noticed in Step 1 | The profile is computed on raw data, so a numeric column with `?` is judged on its real values only. Re-profiling after stage 5 would be tidier, but the disguised-token exclusion already handles the case that mattered. |
