FROM node:22-bookworm-slim AS ui-builder

WORKDIR /app/dermo-chatbot/ui

COPY dermo-chatbot/ui/package*.json ./
RUN npm ci

COPY dermo-chatbot/ui ./
RUN npm run build


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/dermo-chatbot

COPY dermo-chatbot/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY dermo-chatbot ./
COPY --from=ui-builder /app/dermo-chatbot/ui/dist ./ui/dist

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.fastapi_app:app --host 0.0.0.0 --port ${PORT:-10000}"]
