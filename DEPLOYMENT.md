# Deployment

## Render

This repo is prepared for a single-service Render deployment.

### What gets deployed

- FastAPI API
- React UI built during Docker image build
- UI served by FastAPI from `dermo-chatbot/ui/dist`

### Steps

1. Push this repo to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Connect the GitHub repo `enkaranfiles/dermo-ai`.
4. Render will detect [`render.yaml`](/Users/hodor/PycharmProjects/dermo-ai/render.yaml).
5. Set `ANTHROPIC_API_KEY` in the Render dashboard.
6. Deploy.

### Health check

- `GET /health`

### Local Docker test

```bash
docker build -t dermo-ai .
docker run --rm -p 10000:10000 -e ANTHROPIC_API_KEY=your_key dermo-ai
```

Then open:

- `http://127.0.0.1:10000`
- `http://127.0.0.1:10000/health`
