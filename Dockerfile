FROM python:3.9 as base

RUN pip install poetry --upgrade
COPY concept_hierarchy concept_hierarchy
RUN chmod +x concept_hierarchy
COPY poetry.lock .
COPY pyproject.toml .

RUN poetry install --no-root --only main
RUN poetry run pip install tensorflow=="2.10.0"

ENTRYPOINT /bin/bash