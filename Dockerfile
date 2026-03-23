FROM python:3.11-slim

# Install the Docker CLI so docker_manager.py can call `docker start/stop`.
# We only need the CLI, not the daemon.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .
COPY docker_manager.py .
COPY audit.py .
COPY plugins/ ./plugins/

EXPOSE 8080
CMD ["python", "proxy.py"]
