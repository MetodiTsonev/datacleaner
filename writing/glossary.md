# Терминологичен речник / Terminology glossary

**Purpose.** There is no settled Bulgarian rendering for much of this vocabulary. Deciding once,
here, prevents inconsistency across ~75 pages — and this table becomes an appendix.

**Convention.** Where a term is conventionally left in English in Bulgarian technical writing, the
English form is given in brackets and should be used at first mention, e.g.
*блокиране (blocking)*. Thereafter use the Bulgarian form alone.

**Status:** draft — review before Chapter 2 prose begins. Mark disputed rows with ⚠.

---

## Общи / General

| English | Български | Note |
|---|---|---|
| data preparation | подготовка на данни | title term |
| data processing | обработка на данни | title term |
| data quality | качество на данните | |
| data-centric AI | ориентиран към данните ИИ (data-centric AI) | keep English at first mention |
| life cycle | жизнен цикъл | |
| pipeline | конвейер (pipeline) | ⚠ "pipeline" is widely kept untranslated; recommend keeping English throughout for readability |
| modular | модулен | |
| plugin | разширение (plugin) | |
| baseline | базово ниво / еталон | use *еталон* for a comparator, *базово ниво* for a metric floor |
| ablation | ablation (изключване на компоненти) | ⚠ no accepted Bulgarian; explain once, then use English |
| dataset | набор от данни | |
| corpus | корпус | |
| batch | партида | |
| record | запис | |
| feature | признак | **not** "характеристика" — be consistent |
| target variable | целева променлива | |

## Липсващи стойности / Missing data

| English | Български | Note |
|---|---|---|
| missing value | липсваща стойност | |
| missingness | липсващност | ⚠ awkward but needed as a noun; alternative: *наличие на липсващи стойности* |
| disguised missing value | скрита липсваща стойност | e.g. `?`, `-999`, `N/A` |
| missing data mechanism | механизъм на липсващите данни | |
| MCAR — missing completely at random | напълно случайно липсващи данни (MCAR) | |
| MAR — missing at random | случайно липсващи данни (MAR) | |
| MNAR — missing not at random | не случайно липсващи данни (MNAR) | |
| monotone pattern | монотонен шаблон | |
| missingness pattern | шаблон на липсващите стойности | |
| complete-case analysis | анализ само на пълните записи | a.k.a. listwise deletion / *заличаване по редове* |
| imputation | импутация | define once as *запълване на липсващи стойности* |
| single imputation | единична импутация | |
| multiple imputation | множествена импутация | |
| chained equations (MICE) | верижни уравнения (MICE) | |
| Rubin's rules | правила на Rubin | |
| within-imputation variance | вътрешноимпутационна дисперсия | Ū |
| between-imputation variance | междуимпутационна дисперсия | B |
| total variance | обща дисперсия | T |
| confidence interval coverage | покритие на доверителния интервал | the headline E1-M metric |
| sensitivity analysis | анализ на чувствителността | |
| delta-adjustment | delta-корекция | ⚠ keep English "delta" |
| tipping point | точка на обръщане | |
| Little's MCAR test | тест на Little за MCAR | |
| Wald-type statistic | статистика от тип Wald | **not** "likelihood-ratio" — see `review-01.md` A2 |
| degrees of freedom | степени на свобода | |

## Дубликати / Duplicates

| English | Български | Note |
|---|---|---|
| duplicate record | дублиран запис | |
| exact duplicate | точен дубликат | |
| near-duplicate | приблизителен дубликат | central term — be consistent |
| deduplication | отстраняване на дубликати | |
| entity resolution | отъждествяване на записи | ⚠ alt: *разпознаване на обекти* |
| record linkage | свързване на записи | |
| blocking | блокиране (blocking) | |
| shingling | разбиване на к-грами (shingling) | |
| shingle / k-gram | к-грам | |
| MinHash | MinHash (минимизиращо хеширане) | keep English name |
| LSH — locality-sensitive hashing | локално чувствително хеширане (LSH) | |
| banding | разделяне на ленти (banding) | |
| band / row (in LSH) | лента / ред | b bands of r rows: *b ленти по r реда* |
| signature | подпис | |
| candidate pair | кандидат-двойка | |
| Jaccard similarity | коефициент на Жакар | |
| containment | вложеност | asymmetric measure — see `review-01.md` C1 |
| union-find | обединяване-намиране (union-find) | |
| transitive closure | транзитивно затваряне | |
| cluster | кластер | |
| survivorship rule | правило за избор на оцеляващ запис | |
| canonical record | каноничен запис | |
| reduction ratio (RR) | коефициент на редукция | |
| pair completeness (PC) | пълнота на двойките | |
| pair quality (PQ) | качество на двойките | |
| sorted neighbourhood | сортирано съседство | baseline method |

## Признаци / Features

| English | Български | Note |
|---|---|---|
| feature engineering | конструиране на признаци | ⚠ alt: *инженеринг на признаци*; recommend *конструиране* |
| feature generation | генериране на признаци | |
| feature selection | селекция на признаци | |
| one-hot encoding | едно-горещо кодиране (one-hot) | keep English at first mention |
| target encoding | кодиране по целевата променлива | |
| frequency encoding | кодиране по честота | |
| binning / discretisation | дискретизация | |
| scaling | скалиране | |
| standardisation | стандартизиране | |
| skewness | асиметрия | |
| mutual information | взаимна информация | |
| variance threshold | праг на дисперсията | |
| Mincer earnings function | уравнение на заплащането на Mincer | the economics contribution |
| years of schooling | години на образование | |
| potential experience | потенциален трудов опит | age − schooling − 6 |
| interaction term | взаимодействащ член | |

## Валидация и инфраструктура / Validation and infrastructure

| English | Български | Note |
|---|---|---|
| data profiling | профилиране на данни | |
| detector | детектор | |
| data contract | договор за данни | |
| schema | схема | |
| schema evolution | развитие на схемата | |
| validation | валидация | |
| quarantine | карантина | for rejected records |
| scorecard | оценъчна карта | interface, **not** evidence — see `review-01.md` C2 |
| completeness | пълнота | quality dimension |
| uniqueness | уникалност | quality dimension |
| validity | валидност | quality dimension |
| consistency | съгласуваност | quality dimension |
| recipe | рецепта | the serialised pipeline |
| operation | операция | |
| fitted state | напаснато състояние | |
| fit / transform | напасване / преобразуване | |
| lineage / provenance | произход / проследимост | |
| train/test split | разделяне обучение/тест | |
| data leakage | изтичане на информация | |
| out-of-fold | извън дяла | |
| cross-validation | кръстосана валидация | |
| stratified | стратифициран | |
| determinism | детерминираност | |
| reproducibility | възпроизводимост | |
| idempotency | идемпотентност | f(f(x)) = f(x) — **not** the same as determinism |
| checksum | контролна сума | |
| drift | дрейф | ⚠ alt: *отклонение в разпределението*; reviewed in Ch. 2, not implemented |
| medallion architecture | слоева архитектура (medallion) | described as design; implemented as a directory convention |
