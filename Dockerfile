FROM node:22-bookworm-slim AS ui-builder

WORKDIR /app/dermo-chatbot/ui

COPY dermo-chatbot/ui/package*.json ./
RUN npm ci

COPY dermo-chatbot/ui ./
RUN npm run build


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app/dermo-chatbot

# System deps for Pillow and OpenCV (used by torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dermo-chatbot/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY dermo-chatbot ./
COPY --from=ui-builder /app/dermo-chatbot/ui/dist ./ui/dist

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.fastapi_app:app --host 0.0.0.0 --port ${PORT:-10000}"]
