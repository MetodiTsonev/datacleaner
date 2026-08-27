# Critical Review 01 — findings against `concept.md` v2

**Date:** 2026-08-14
**Method:** two independent adversarial passes — a feasibility/scope review and a hostile-examiner technical review — plus reconciliation.
**Status:** findings accepted. `concept.md` v2 must be revised to v3 before any code is written.

---

## A. Thesis-threatening problems (fix before anything else)

### A1. Feature engineering has no owner — a regression against a mandatory requirement

The annotation names **three** mandatory theoretical topics: missing values, duplicate removal, **feature engineering**. v2 nominates **two** from-scratch algorithms and says they were "chosen so each lands on one of the annotation's mandatory theoretical topics" — leaving the arithmetic gap visible in the document.

Feature engineering in v2 has: no contribution in the list of seven, no UI page, no experiment, no ablation, and one mention in the module tree. `ideas.md` §3 Module 5 had a concrete FE specification; v2 dropped it and replaced it with nothing.

Worse, the corpus choice compounded it — news text offers almost no numeric feature engineering, and the tabular corpus's interesting cases (lags, growth rates, ratios, per-capita normalisation) were never mentioned.

**Verdict:** the single most likely cause of a bad grade in the current design. FE needs a flagship, a contribution, a UI surface and an ablation experiment.

### A2. The missingness algorithm claims things that are provably impossible, on a corpus where the theory does not apply

Multiple independent errors, several of them load-bearing:

- **The statistic is misdescribed.** Little's (1988) d² is a Wald-type quadratic form on pattern-wise observed means, **not** a likelihood-ratio statistic as v2 §3.2 states. Degrees of freedom (Σⱼpⱼ − p) are never stated — exactly where re-implementations go wrong.
- **Non-rejection is treated as evidence for MCAR.** v2 step 5 routes "MCAR + low rate → simple imputation", i.e. accepting the null. Only rejection is a valid inference.
- **MAR vs MNAR is not identifiable from observed data.** For any MAR model there exists an MNAR model with an identical observed-data likelihood (Molenberghs et al., 2008 — "every MNAR model has a MAR bodyguard"). v2 step 3 promises to distinguish them via classifier AUC. High AUC is consistent with *both*, and especially with self-masking MNAR when columns are correlated. The method has no discriminating power over the hypotheses it adjudicates.
- **The MNAR heuristics are backwards or vacuous.** "Missingness correlated with the target" is, if the target is observed, the *definition of MAR*. Monotone patterns are a descriptive property of R and carry no mechanism information. "Non-MCAR + low MAR-predictability → MNAR" is argument from ignorance built on two uninformative signals.
- **The defensible MNAR treatment is absent.** The literature's answer to suspected MNAR is *sensitivity analysis* — pattern-mixture models, delta-adjustment, tipping-point analysis. A missingness indicator does not correct MNAR bias.
- **World Bank WDI is a category-error minefield.** A large share of cells are *undefined*, not *missing*: the country did not exist that year (USSR pre-1992, South Sudan pre-2011), the series had a definitional start date or was discontinued, or the indicator is not collected for that country by design. There is no counterfactual value to impute. Rubin's taxonomy describes a stochastic response mechanism over a well-defined complete-data matrix; applying it to structural non-existence is a category error.
- **Panel data breaks Little's test irreparably.** It assumes i.i.d. draws from one MVN. WDI has serial dependence (near-unit-root series), between-country heterogeneity spanning orders of magnitude, and cross-sectional dependence (global shocks). Critically, **WDI missingness patterns are essentially country-group identifiers** — OECD reports series X, low-income countries do not — so pattern means differ because the *composition of countries* differs, not because missingness depends on values. The test cannot separate composition from mechanism. It will reject, and the rejection is uninterpretable.
- **Σ̂ will be singular.** WDI contains near-exact linear dependencies (GDP / GNI / GDP per capita / GDP PPP; total-male-female decompositions; shares summing to 100). EM under MVN has no unique solution and (Σ̂ⱼᵒᵇˢ)⁻¹ is undefined. No ridging or collinearity pre-screen was specified.
- **MVN is indefensible on WDI** — log-normal, bounded-[0,100], count-like and heavy-tailed series. d² rejects for non-normality, and nothing distinguishes that from non-MCAR.
- **MICE ≠ `IterativeImputer`.** MICE is *multiple* imputation: m > 1 datasets, pooled by Rubin's rules. sklearn's `IterativeImputer` is single imputation. v2 names Rubin as the theoretical anchor, implements single imputation, and never mentions m or pooling.
- **The evaluation criterion contradicts the theory.** Per-cell RMSE selects conditional-mean imputation, which shrinks variances and distorts covariances — exactly the failure multiple imputation exists to avoid. Chapter 2 would present a theory that Chapter 5 evaluates against a criterion the theory rejects.
- **The economic argument is self-defeating.** "Weak statistical systems report less" — but statistical capacity is *predictable from observed WDI columns* (GDP per capita, governance indicators, IDA status). If so, the mechanism is MAR, destroying the stated reason for choosing WDI. Also uncited and untestable.

**Verdict:** WDI must be dropped for the missingness work, and the algorithm reframed around what is actually identifiable.

### A3. Leakage is baked into the architecture, contradicting contribution #5

v2 §2.1 places imputation and outlier treatment in **silver**, and materialises the train/test split in **gold**. Imputers, outlier bounds *and the mechanism diagnosis itself* are therefore fitted on data that includes the test rows. This directly contradicts E4 ("no test-set statistic ever reaches a fitted transformer"), contribution #5, and `ideas.md`'s "strict fit-on-train-only discipline".

A one-line architecture change fixes it: **split first, then fit everything downstream of the split.**

---

## B. Feasibility — the plan is ~9–11× oversized

Independent bottom-up estimate of v2 §7's 17 build steps: **≈ 60 full developer-days**, ~65–75 with solo-integration tax. Available: ~20 calendar days at 2–3 productive hours alongside a full-time job ≈ **6–8 full dev-days** — while also writing 70–80 Bulgarian pages, itself 18–25 days of work.

The plan fails around step 9 of 17, with no evaluation, no UI and no thesis.

### Three biggest sources of unnecessary complexity

1. **The platform tier itself** — medallion implementation, contracts subsystem, quarantine subsystem, column-level lineage, drift module, incremental runs, DuckDB, SQLite registry, batch manifests. ~22 dev-days that produce **architecture diagrams, not thesis findings**. The diagrams can be *designed* in Chapter 3 in a day.
2. **Two live-ingested corpora and three network connectors.** Making streaming primary makes ingestion, scheduling, batching, incrementality and drift all load-bearing. ~11 dev-days, and the annotation asks for none of it.
3. **The evaluation surface** — 4 experiments × 4 corruption rates × 2 corpora × 3 models × 3 ground-truth regimes, plus 70% coverage. E1 alone has 30+ measured cells, and Chapter 5 is budgeted at 12–14 pages. Measurements are being built that there is no room to present.

### Component verdicts

| Component | Verdict |
|---|---|
| Medallion bronze/silver/gold | **SIMPLIFY** → three directories (`raw/`, `interim/`, `processed/`) as a naming convention. Describe the pattern properly in Ch.3; state the prototype implements it as a convention. 0.25d not 4d |
| DuckDB | **CUT.** pandas + pyarrow suffices at this scale. One sentence in Ch.4 as the scaling path |
| SQLite registry | **CUT** → `datasets.yaml` + one JSON per run in `runs/` |
| Column-level lineage | **SIMPLIFY** → op-level provenance recorded in the run JSON (already implied by the recipe), rendered as Mermaid. Requires no changes to any op. Contribution survives |
| Drift module | **CUT to one function** — `compare_profiles()` returning PSI/KS per column + schema diff. Cut vocabulary drift, cut the drift experiment, cut the UI page |
| Quarantine | **KEEP, trivially** — validation returns `(valid, invalid_with_reason)`; write `rejected.csv`. It is the visible form of "validating" |
| Contracts + schema diff | **SIMPLIFY** → Pandera `infer_schema` → YAML → `validate`. Schema diff = compare two column→dtype dicts. No contract DSL |
| Recommendation rule engine | **KEEP.** Cheapest high-value component; it *is* the automation contribution |
| HTML report | **KEEP, minimal.** One template. Also supplies appendix/screenshot material |
| Streamlit 9 pages | **SIMPLIFY to 4.** UI is mandated (R5) so it cannot be cut, but 9 pages is 5d and 4 thin pages is 1.5d |
| Ad-hoc upload path | **PROMOTE to primary.** Cheapest entry point, and what the annotation literally describes ("takes raw data sets") |
| Two corpora | **KEEP the pair** — one text corpus for dedup, one tabular for missingness/FE. One corpus cannot carry all three mandatory topics |
| GDELT connector | **CUT.** Highest cost-to-value ratio in the plan; raw 15-minute files are headerless multi-column TSVs with packed sub-fields, ~2–3d. Mention in Ch.2 and future work |
| RSS collector | **DEMOTE** to an optional ~30-line script, depended on by nothing |
| Little's MCAR test | **KEEP but descope and reframe** — see A2. Validate against a reference implementation from day one |
| MinHash + LSH | **KEEP.** Descope pair scoring; keep the sorted-neighbourhood baseline |
| Corruption harness | **KEEP.** Highest-leverage single component. Descope to missingness mechanisms, duplicate injection with 3 perturbation types, outliers, typos |
| E1–E4 | **KEEP E1, E2, E4. CUT E3** (it needs the deleted platform tier) |
| 70% coverage | **CUT the number** → ~50% scoped to `ops/`, `recipe/`, `algorithms/`, stated explicitly in Ch.4 |

### Schedule traps — small-looking work that reliably explodes

1. **GDELT raw files** — 2–3 days disguised as a download loop.
2. **Little's test with EM** — the trap is that a wrong implementation *runs fine and prints a plausible p-value*. Validate against a published worked example or R's `naniar::mcar_test` from day one, not at the end.
3. **Column-level lineage** — means touching and re-testing all ~15 operations. Classic retrofit that doubles the cost of what it's retrofitted into.
4. **Streamlit session state** — Streamlit re-runs the whole script on every interaction; any stateful multi-step flow costs a day of `st.session_state` design. Every developer loses this day exactly once.
5. **"70% coverage"** — the last 20 points cost more than the first 50 and produce no thesis value.
6. **"Byte-identical output"** — see C3. A day spent discovering it's impossible, then weakening the claim anyway.
7. **Hand-labelling a few hundred pairs** — 3–6 hours plus a labelling protocol plus inter-annotator caveats you cannot address with one annotator. Cut.
8. **Language detection on titles** — noisy on 8–12 word strings; visible errors in demo screenshots; no thesis payoff. Cut.
9. **Figures → thesis** — Plotly needs `kaleido`; kaleido breaks; fonts don't match the template; every figure needs a Bulgarian caption and a numbered reference. **~0.25 days per final figure.** Cap at 12–15 figures and decide the list before generating any.
10. **Bulgarian terminology** — there is no settled Bulgarian for "near-duplicate", "banding", "data contract", "fitted state". Build a ~40-term glossary on day 1; it becomes an appendix.
11. **The unknown faculty template** — retrofitting into 70 finished pages is 1–2 lost days.
12. **The plugin abstraction** — reduce to a single `REGISTRY: dict[str, type]` with a `@register` decorator. Satisfies the claim in 2 hours instead of 1.5 days.

### The RSS collector is the plan's only unrecoverable single point of failure

Volume is probably fine (~12–24k items over 20 days). The real problems:

- **Corpus length is capped by wall-clock, not effort.** A collector bug on day 3 noticed on day 12 costs nine days that cannot be recovered by working harder. Nothing else in the plan behaves this way.
- **Organic duplicates have no labels**, so E1 falls back on injected duplicates anyway — the live corpus contributes almost nothing to the headline measurement.
- **The freeze deadline collides with the code deadline** — the corpus must freeze around day 17, yielding 14–16 usable days.
- **Titles + summaries are too short for reliable shingling** (see C1).

**Invert the dependency:** a static offline corpus is primary from day 1; the collector is a bonus that, if it works, becomes a short "live case study" in Chapter 5 — which is a *stronger* defense moment than the streaming architecture, because it demonstrates domain-neutrality with real evidence. If it produced nothing, two paragraphs get deleted.

---

## C. Technical corrections needed regardless of scope

### C1. MinHash / LSH

- **The S-curve `1 − (1 − sʳ)ᵇ` is correct**, but two things must be added or a reviewer assumes ignorance: the constraint **h = b·r** (so sweeping (b, r) changes signature length and requires recomputing signatures — a cost never budgeted), and the approximate threshold **s\* ≈ (1/b)^(1/r)**.
- **Attribution.** Broder contributed shingling + min-wise independent permutations. The **banding/S-curve construction is LSH from Indyk & Motwani**, popularised by Leskovec/Rajaraman/Ullman. Chapter 4 must not attribute banding to Broder.
- **Jaccard is the wrong similarity for the motivating example.** Truncation and added boilerplate create length asymmetry, which Jaccard penalises severely. **Containment** (asymmetric) is appropriate, and MinHash-based containment estimation is a known technique. As specified, the algorithm is blind to its own motivating case.
- **State whether you used h independent hash functions or bottom-k/one-permutation hashing** — the banding formula does not transfer to the latter.
- **Short text is a real methodological weakness.** 15–45 words → ~15–40 shingles. True Jaccard is coarsely quantised over ~30 values, so the S-curve is evaluated at lumpy, unstable s, and one rewritten headline moves s by a large discrete step. Publisher boilerplate can dominate a 30-shingle set, making *every item from one feed a near-duplicate of every other*. Boilerplate induction is itself a research problem normally done from full HTML across a site — explicitly ruled out here.
- **Bucket-size skew** is the classic LSH failure: one heavily syndicated release produces a bucket whose internal verification is quadratic, collapsing the promised reduction on exactly the data the thesis is built on. No bucket capping specified.
- **Pair metrics vs cluster actions.** Union-find takes a transitive closure over noisy edges → chaining merges non-duplicates via intermediates. v2 measures *pairs* and acts on *clusters*. Cluster-level metrics (B-cubed, ARI, or at minimum cluster-size distribution and purity) are required and absent — survivorship errors are what actually corrupt the output.
- **Pair quality (PQ) is missing** alongside reduction ratio and pair completeness. RR and PC alone are jointly gameable. RR is also dominated by trivially high values (0.9999+) for any scheme, so "order-of-magnitude reduction" is an unimpressive headline the baseline also achieves.
- **"Exact-match dedup finds none of them" is asserted and probably false** for verbatim wire copy after normalisation. Make it a measurement, not a claim.
- **The tuned similarity and the decision similarity are different objects.** (b, r) is tuned for a Jaccard threshold, but accept/reject uses a field-weighted composite. The S-curve does not describe the operating point of the evaluated system.
- **Cross-batch dedup is unspecified.** Syndication delay is hours to days; within-batch dedup misses most duplicates.

### C2. The quality scorecard is not evidence

- ISO/IEC 25012 defines quality *characteristics* and deliberately prescribes no aggregation function. Any "quality rose from 42 to 87" is a function of an arbitrary weight vector. Expect: *"recompute with equal weights and show the conclusion holds."*
- Linearly summing completeness %, uniqueness % and validity % asserts a constant marginal rate of substitution between them, with no basis.
- Compensatory aggregation hides fatal defects — catastrophic validity can still score 70.
- **It is circular as evidence.** The components are exactly the defects the pipeline repairs; "score after > score before" is a tautology. The only non-circular evidence of improvement is E2's downstream utility.

**Verdict:** demote the scorecard from *evidence* to *interface*. Report the vector, not just the scalar; use a non-compensatory rule or a veto for fatal defects.

### C3. "Byte-identical output" is unachievable and mislabelled

- **Terminology:** same input → same output is *determinism/repeatability*, not **idempotency** (f(f(x)) = f(x)). The idempotency question — does running twice duplicate rows? — is genuinely interesting and was not tested.
- **Concrete breakers:** pyarrow writes a `created_by` version string and pandas schema metadata; row-group boundaries depend on chunk size; compression output depends on codec version (gzip embeds mtime); **`PYTHONHASHSEED` / set-iteration order** silently reorders encoder vocabularies and one-hot columns; float non-associativity under BLAS/DuckDB parallelism changes imputation fill values in the last bits; `IterativeImputer`/`KNNImputer` depend on `random_state`, tolerance and BLAS; run metadata embeds timestamps.
- **Fix:** define reproducibility as a **canonical-form content hash** (rows sorted by explicit key, fixed column order and dtypes, fixed float rounding, metadata excluded), pin the environment with a lockfile, set `PYTHONHASHSEED=0`, force single-threaded numeric libraries, and never let unordered `set` reach output.

---

## D. Internal contradictions found

| # | Contradiction |
|---|---|
| D1 | Imputation in silver, split in gold → leakage in the architecture diagram, one layer above the test that claims to exclude it (= A3) |
| D2 | Out-of-time split materialised in §2.1, but E2 says "cross-validated" — random k-fold on a panel leaks across time and within country |
| D3 | **Two competing imputation rule sets with no precedence** — `ideas.md`'s rate/skew rules (declared "still valid") vs v2's mechanism-based rules. A 3%-missing MNAR column gets contradictory verdicts |
| D4 | The retained `ideas.md` rules are corpus-inapplicable — "`NA` = no garage", high-cardinality target encoding, explicit `Missing` category — written for Ames/Adult, meaningless for a numeric panel or text |
| D5 | `ideas.md`'s supersession banner names a corpus (Online Retail II, Eurostat) that v2 does not adopt. Three different corpora described across two documents |
| D6 | The two supersession lists disagree about what survives |
| D7 | The effort claim survives the deletion of its measurement — E3's effort study was cut and Ch.1's ROI modelling removed, yet "a fraction of the human effort" and "human upper bound" persist. **Effort is now measured nowhere** |
| D8 | "No NLP" vs. specified language detection, TF-IDF, and GDELT themes (themes are GDELT's own NLP over full text) |
| D9 | Invariant "anything the app can do, the CLI can do" is violated by the module list — pages for Duplicates, Lineage, Drift, Export have no CLI verbs |
| D10 | The contract is "the gate between bronze and silver" but silver is the first *typed* layer — type/range/category rules cannot be evaluated on raw XML strings. The order of operations as drawn is impossible |
| D11 | Bronze is "never modified" but WDI "revisions appear between pulls" and silver is "one row per valid record" — successive batches deliver conflicting values for key (country, indicator, year) with no merge/upsert semantics defined |
| D12 | Drift detection detects the shift that invalidates the fitted state that E3 requires be reused. No refit policy |
| D13 | Reproducibility requires shipping a frozen hashed corpus; licensing forbids redistributing feed text; §9 asks about a public repo. Unresolved trilemma |
| D14 | A document titled "locked decisions" whose §9 lists **the assignment text itself** as still missing — and R1–R5 are inferred from a truncated sentence, then referred to elsewhere as established fact |
| D15 | "Pandera **or** pydantic" in an authoritative document — not substitutes (one validates dataframes, the other objects) |
| D16 | "Two algorithms from scratch" is inaccurate — EM for MVN with arbitrary missingness is a third non-trivial implementation, and PSI a fourth |

---

## E. Evaluation validity

- **E1 ground truth is contaminated and circular.** The corruptor and detector share an author, with no held-out corruption family. MNAR is not one thing (self-masking, threshold censoring, shifted-logit, pattern-mixture) and detectability differs by family. **Remedies:** use third-party corruption tooling (`pyampute`, `jenga`, CleanML setups) so the generator is not yours; hold out at least one mechanism family; fix thresholds before seeing the sweeps; include at least one real dirty/clean pair you did not create.
- **Injecting missingness into an already-40–70%-missing matrix invalidates the label** — the result is a *mixture* of pre-existing and injected mechanisms. Restricting to a complete sub-panel evaluates the least-MNAR, least-representative slice. Genuine dilemma; state which horn you take.
- **Injected outliers are indistinguishable from real ones** (hyperinflation, war-year collapses), so precision against injected-only truth *penalises correct detection of real anomalies*.
- **E2's biggest defect: the treatment changes the evaluation set.** In the raw arm, near-duplicates split across train/test → contamination → **inflated** accuracy. In the pipeline arm they're removed → harder task. The comparison confounds data quality with test-set composition, and the bias may run *against* the headline. **Fix: one fixed, deduplicated held-out evaluation set; vary only training data across arms.**
- **Feed `<category>` labels are noisy, publisher-selected and leaky** — the spec itself lists missing `category` as a target pathology, then uses it as labels. Categories are publisher-specific free text needing a harmonisation mapping the author authors. Category correlates with source domain, so residual boilerplate lets the classifier identify the *publisher*, meaning the arm that strips boilerplate best would *lose*.
- **GDELT themes as labels are worse** — produced by GDELT's taggers over full article text the classifier doesn't have. Predicting them measures distillation from a stronger upstream tagger, not data quality.
- **The manual arm is unblinded and self-serving.** The author designs the corruption, the automation and the metric, then plays the comparator. n = 1, no protocol, unreproducible, and the effort side was deleted. **Fix:** pre-register a time-boxed protocol with a published decision log, and add *external* baselines — naive pandas, a Great Expectations workflow, ydata-profiling-guided cleaning, CleanML configurations.
- **Statistics are inadequate** — std across dependent CV folds is not a valid uncertainty estimate for comparing pipelines; no significance testing, no multiplicity correction across 4 arms × 3 models × 2 corpora.
- **No "tune the model instead" arm**, so the data-centric claim (R2) is only half-tested.
- **"Zero leak-through" is unfalsifiable puffery**, and the **false-quarantine rate is not measured at all** — for news, where publishers legitimately evolve schemas, that is the failure mode that silently destroys the corpus.
- **Drift sensitivity is measured under a false null** — needs the alarm rate on consecutive *un-injected* batches, since news has strong diurnal and weekly seasonality. PSI's 0.1/0.25 thresholds are credit-scoring folklore with no null distribution; KS is invalid with ties and over-sensitive at large n.
- **Coverage is not a result.** It measures execution, not correctness, and invites "which 30% is untested — is it the two algorithms?"
- **Lineage is asserted, not observed** — the graph is built from each operation's self-declaration, so a buggy op reports clean lineage. No verification strategy for contribution #5.

---

## F. Other defence hazards

1. A textbook S-curve derivation called "the mathematics of the thesis" invites *"what did you derive?"*
2. Chapter 2 at 18–22 pages for eight-plus topics ≈ 2.5 pages each; a mandatory topic getting two pages is what committees mark down.
3. "Modern methods" is under-delivered — scoping missing values "до MICE/KNN" answers a 2026 thesis with a 2001 canon. The *theoretical* chapter must survey GAIN/MIWAE/VAE imputation, transformer-based entity resolution (Ditto and successors), LLM-assisted cleaning.
4. Drift and "vocabulary drift" over a ~20-day corpus is not a credible temporal study; in-window seasonality dominates.
5. Requirements engineering (functional/non-functional) was dropped from Ch.3 — Bulgarian committees in this discipline expect it, traceable to the задание.
6. **No requirements-traceability table** (R1–R5 → module → chapter → experiment). Its absence is why the FE gap survived into a document labelled authoritative. Cheapest available defence.
7. **GDPR unaddressed** — news articles contain personal data about named individuals being stored, processed and possibly published.
8. Concurrency hazards guaranteed to surface in a live demo: a background poller and Streamlit both touching SQLite and DuckDB (single-writer each); Streamlit's full-rerun model against long-running polls.
9. The mandated UI is step 15 of 17, with Chapter 4 screenshots owed.
10. Step 0 admits network egress was unverified — the corpus, connectors and every Chapter 5 number sit behind an unverified assumption, recorded as a bullet under "Practical notes".
11. **The economic contribution is effectively nil** after the ROI chapter was cut, and the one economic argument (WDI MNAR) is uncited, untestable and self-defeating. For a programme in AI in Economics, expect the question.
12. `ideas.md` remains in the repo with a factually wrong supersession banner and a duplicate open-questions list.

---

## G. Highest-priority fixes, in order

1. **Split before fitting.** Move imputation and outlier treatment below the train/test split. One-line architecture change that rescues the leakage contribution.
2. **Give feature engineering a flagship, a contribution, a UI surface and an ablation.**
3. **Drop WDI for the missingness work** and rewrite the missingness module around what is identifiable: a structural-vs-stochastic cell partition, an MCAR test with the correct statistic and stated df (rejection-only inference), proper *multiple* imputation with Rubin's pooling, and **sensitivity analysis** in place of MNAR "detection".
4. **Fix E2** — one fixed deduplicated held-out evaluation set, non-author baselines, real labels rather than feed categories.
5. **Delete the platform tier as implementation**; keep it as Chapter 3 design with an alternatives discussion.
6. **Make both corpora static and offline**; the RSS collector becomes an optional bonus.
7. **Replace "byte-identical" with a canonical-form content hash** and rename the property.
8. **Demote the scorecard from evidence to interface.**
9. **Add a requirements-traceability table** to Chapter 3.
10. **Get the faculty template and the full assignment text.** Still blocking, still free.
