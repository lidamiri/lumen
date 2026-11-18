FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 1000 lumen && mkdir -p /data && chown lumen:lumen /data

USER lumen

ENV LUMEN_DATA_DIR=/data

EXPOSE 8000

CMD ["lumen", "serve", "--host", "0.0.0.0", "--port", "8000"]
