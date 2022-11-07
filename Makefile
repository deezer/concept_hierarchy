.PHONY: usage check clean run

CODE_DIRECTORY := concept_hierarchy

usage:
	@echo "Available commands:\n-----------"
	@echo "	check		Checks coding conventions using multiple tools"
	@echo "	clean		Runs black and isort to auto format your code"

check:
	poetry run pyflakes $(CODE_DIRECTORY) || exit 1
	poetry run black --check $(CODE_DIRECTORY) || exit 1
	poetry run isort --check $(CODE_DIRECTORY) || exit 1
	@echo "\nAll is good !\n"

clean:
	poetry run black $(CODE_DIRECTORY)
	poetry run isort $(CODE_DIRECTORY)

run:
	poetry run python 1_CAV_train.py
	poetry run python 2_CAV_aggregate.py
	poetry run python 3_compute_projections.py
	poetry run python 4_hierarchy.py


