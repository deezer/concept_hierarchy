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

docker:
ifndef SCRIPT
	@printf "\n script name needed (e.g SCRIPT=1_CAV_train) !\n\n"
	exit 1
endif

	docker build . -t concept_hierarchy
	docker run -ti -v data:/data -v results:/results -v weights:/weights concept_hierarchy sh -c "poetry run python concept_hierarchy/${SCRIPT}.py"
