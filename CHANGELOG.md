# Changelog

## Unreleased

- Split the original local pipeline into `backend/` and `frontend/` projects while preserving the PDF -> GROBID/LightOCR -> JSON-LD ModelCard flow
- Added Poetry environments, Dockerfiles, `.dockerignore` files, and Docker Compose services for Ollama/model init, GROBID, RabbitMQ, backend, worker, and frontend
- Added FastAPI job endpoints: `POST /launch-job/` and `GET /launch-job/job-status/{job_id}` with persisted `status.json` files under `DATA_DIR/jobs`.
- Added RabbitMQ job publishing and a worker with dummy mode for smoke tests and real `PDFHandler` mode for full pipeline runs
- Moved backend code to package-style imports and environment-driven config, including shared `DATA_DIR` support for Docker bind mounts.
- Hardened the real worker path with 600s external timeouts, explicit PDF/GROBID/OCR failures, corrected dataset extraction inputs, and worker-visible `PDFHandler` progress logs
- Added backend/frontend `.dockerignore` files and kept only a small runtime data sample