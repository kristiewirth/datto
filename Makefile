.PHONY: all docs clean

docs:
	(cd docs; make clean; sphinx-build -b html source build -E -a)

test:
	pytest -v -p no:warnings

publish:
	poetry update; make docs; poetry build; poetry publish