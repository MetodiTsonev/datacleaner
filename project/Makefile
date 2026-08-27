PY := ../.venv/bin/python

.PHONY: app test lint fix sample
app:    ; PYTHONPATH=. $(PY) -m streamlit run app.py
test:   ; $(PY) -m pytest
lint:   ; $(PY) -m ruff check src app.py tests
fix:    ; $(PY) -m ruff check --fix src app.py tests
sample: ; $(PY) ../make_dirty_sample.py
