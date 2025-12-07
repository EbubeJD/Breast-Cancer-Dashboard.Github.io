# ---------- Stage 1: build dependencies ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install

# Build tools for scientific Python wheels (not kept in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ---------- Stage 2: runtime image ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy your app code + data
COPY . .

EXPOSE 8050

# Run with Gunicorn in production
# app:server → module "app.py", WSGI callable "server"
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]
