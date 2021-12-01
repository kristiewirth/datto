.PHONY: all docs clean

docs:
	(cd docs; make clean; sphinx-build -b html source build -E -a -Q)

test:
	pytest -v -p no:warnings