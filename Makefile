.PHONY: install lint test train all clean

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

test:
	python -m pytest tests/ -v --tb=short

train:
	python src/train.py

all: install lint test train

clean:
	rm -rf mlruns mlflow.db __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
