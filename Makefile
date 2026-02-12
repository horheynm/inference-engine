.PHONY: style

style:
	python -m black .
	python -m flake8 .

