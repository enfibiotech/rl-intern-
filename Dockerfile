FROM python:3.11-slim

WORKDIR /app

# System deps for MuJoCo / rendering (optional, safe to skip for classic control)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY . .

# Core RL dependencies
RUN uv run pip install \
    stable-baselines3[extra] \
    gymnasium[classic-control,box2d] \
    arxiv \
    huggingface-hub \
    litellm \
    rich typer python-dotenv \
    --no-cache-dir

ENV RL_INTERN_WORKSPACE=/workspace
RUN mkdir -p /workspace

ENTRYPOINT ["uv", "run", "rl-intern"]
