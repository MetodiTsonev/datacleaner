# Как да се пусне

Проверено от нулата на 2026-08-30, macOS. Отнема около 3 минути.

---

## 1. Python 3.12

```bash
python3 --version
```

Нужен е **3.12 или по-нов**.

> **На тази машина:** Homebrew `python@3.11` е счупен (несъответствие в символите на
> `pyexpat`/`libexpat`) и `import pandas` се проваля с него. Използвайте варианта от
> python.org:
> `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`

## 2. Виртуална среда и инсталиране

От корена на хранилището:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e "project[dev]"
```

Това инсталира pandas, numpy, streamlit и openpyxl — нищо друго. Списъкът със
зависимости е затворен и има тест, който се проваля, ако някой модул внесе нещо извън
него (виж правило 4 в `PLAN.md`).

> Ако `pip` се провали с грешка за SSL сертификат, задайте:
> `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)`

## 3. Пускане на приложението

```bash
.venv/bin/streamlit run project/app.py
```

Отваря се на <http://localhost:8501>.

## 4. Първи път — какво да пробвате

1. В страничната лента изберете **Sample file → `adult-census.csv`**.
2. Отворете раздела **Profile**. Обърнете внимание: „Missing cells pandas reports“ е
   **0**.
3. Отворете **Findings**. Три колони съдържат хиляди **прикрити** липсващи стойности,
   записани като `?`. Контрастът между тези два екрана е най-краткото обяснение защо
   инструментът съществува.
4. В страничната лента изберете **Column to predict → `income`**.
5. Минете през **Plan** (стъпките в ред, с една **изведена**), **Run** (какво е
   променила всяка стъпка), **Features** и **Evidence**.
6. От **Evidence** изтеглете „What was done (Markdown)“ — записът какво е било направено.

За да видите как се държи с непознат файл, изберете **Upload** и подайте собствен CSV
или XLSX.

## 5. Тестовете

```bash
cd project && ../.venv/bin/python -m pytest
```

Около две минути (наборът от преброяването се обработва многократно).

Проверка на кода:

```bash
.venv/bin/ruff check project scripts
```

## 6. Фигурите за тезата

Само за писането на текста, не за самата система:

```bash
.venv/bin/pip install -e "project[figures]"
.venv/bin/python scripts/figures.py            # ~4 минути
.venv/bin/python scripts/figures.py --quick    # 3 разделяния вместо 10
```

Записва в `writing/figures/`. `matplotlib` **не участва в системата** — виж Р13 в
`writing/decisions.md`.

## 7. Демонстрационният файл

`project/data/input/messy-orders.csv` е създаден нарочно и се пресъздава с:

```bash
.venv/bin/python make_dirty_sample.py
```

Всеки дефект в него е планиран и отговаря на конкретна проверка в `project/src/checks.py`.

---

## Ако нещо се обърка

| Симптом | Причина |
|---|---|
| `ModuleNotFoundError: pandas` | `pip install -e "project[dev]"` не е изпълнен, или се използва системният Python вместо `.venv/bin/python` |
| `import pandas` се срива без съобщение | счупеният Homebrew Python 3.11 — виж т. 1 |
| `pip` дава SSL грешка | задайте `SSL_CERT_FILE` — виж т. 2 |
| Streamlit показва грешка **в страницата** | това е нормалният начин на Streamlit да показва изключения; текстът на грешката е в самата страница |
| `ModuleNotFoundError: matplotlib` при `scripts/figures.py` | инсталирайте допълнението `figures` — виж т. 6 |
