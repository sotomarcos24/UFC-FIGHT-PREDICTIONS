FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1. Dependencies first (cached unless they change)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# 2. Application code and trained model
COPY features.py predict.py model.bin ./

EXPOSE 9696

CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]