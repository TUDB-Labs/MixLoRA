#!/bin/bash
black mixlora
black tests
isort mixlora --profile black
isort tests --profile black
flake8 mixlora --show-source --statistics --max-line-length=128 --max-complexity 15 --ignore=E203,W503,E722
flake8 tests --show-source --statistics --max-line-length=128 --max-complexity 15 --ignore=E203,W503,E722
