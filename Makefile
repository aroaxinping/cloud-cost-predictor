.PHONY: setup ingest train validate predict app demo docker lint test clean all help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies with uv
	uv sync --extra dev --extra notebooks

ingest: ## Stream raw SAP data and build VM summaries
	uv run python src/ingest.py

train: ## Execute the model training notebook
	uv run jupyter nbconvert --to notebook --execute notebooks/03_predictive_model.ipynb

validate: ## Validate clean data before prediction
	uv run python src/validate.py

predict: validate ## Run predictions on VM utilization data
	uv run python src/predict.py

app: ## Launch the Streamlit dashboard
	uv run streamlit run app.py

demo: setup predict app ## Full demo: install, predict, launch

docker: ## Build and run in Docker
	docker build -t cloud-cost-predictor .
	docker run -p 8501:8501 cloud-cost-predictor

lint: ## Lint with ruff
	uv run ruff check .

test: ## Run test suite
	uv run pytest tests/ -v

clean: ## Remove caches and virtual environment
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

all: setup ingest train predict ## Full pipeline
