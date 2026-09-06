VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup ingest train predict app docker lint test clean all

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) src/ingest.py

train:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/03_predictive_model.ipynb

predict:
	$(PYTHON) src/predict.py

app:
	$(VENV)/bin/streamlit run app.py

docker:
	docker build -t cloud-cost-predictor .
	docker run -p 8501:8501 cloud-cost-predictor

lint:
	$(VENV)/bin/ruff check .

test:
	$(PYTHON) -m pytest tests/

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

all: setup ingest train predict
