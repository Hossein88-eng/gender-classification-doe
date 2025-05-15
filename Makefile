# Makefile for Gender Classification with DOE

VENV ?= .venv
PYTHON = $(VENV)/bin/python

.PHONY: help venv install train evaluate plots test lint clean

help:
	@echo "Available commands:"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train the CNN model"
	@echo "  make evaluate    - Evaluate model and print classification report"
	@echo "  make plots       - Generate training history plots"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Check code style with flake8"
	@echo "  make clean       - Remove temporary files and caches"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) src/training_utils.py --train

evaluate:
	$(PYTHON) src/training_utils.py --evaluate

plots:
	$(PYTHON) src/training_utils.py --plot

test:
	$(PYTHON) -m unittest tests/test_all.py

lint:
	flake8 --ignore=E501,W504,W503,E203 src/ tests/

format:
	. .venv/bin/activate && .venv/bin/black src/ tests/

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache

all: install format lint test