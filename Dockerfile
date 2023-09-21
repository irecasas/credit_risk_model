FROM python:3.10.12-slim

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV COVERAGE_FILE /app/.coverage

COPY . /app
RUN chmod +x -R /app
RUN apt-get update
RUN apt-get install -y git unzip xvfb gcc python3-dev zip libpq-dev build-essential

RUN groupadd -r app \
    && useradd -r -m -g app app \
    && pip install --upgrade --no-cache-dir pip \
    && pip install --no-cache-dir poetry \
    && pip install poetry-plugin-export

RUN cd /app \
    && poetry export -f requirements.txt -o requirements.txt --with dev \
    && pip install --no-cache-dir -r requirements.txt \
    && chown -R app:app /app

USER app

ENV PYTHONPATH "${PYTHONPATH}:/app/src/:/app/tests/"

WORKDIR /app/src