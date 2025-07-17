# Use official Python image
FROM python:3.13-slim

# Environment variables
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install Poetry and system deps
RUN apt-get update && apt-get install -y curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency files from your repo root (on host) to container
COPY pyproject.toml poetry.lock ./

# Install Python dependencies with Poetry
RUN poetry install --no-root

# Copy app code (Python source files in app/)
COPY app ./app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit pointing to app/main.py
CMD ["poetry", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
