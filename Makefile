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

train:
	poetry run python 1_CAV_train.py

aggregate:
	poetry run python 2_CAV_aggregate.py

compute_projection:
	poetry run python 3_compute_projections.py

hirerarchy:
	poetry run python 4_hierarchy.py

docker:
	docker build . -t concept_hierarchy
	docker run -ti --rm -v data:/data -v results:/results -v weights:/weights concept_hierarchy bash


