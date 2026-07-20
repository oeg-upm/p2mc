# Changelog

## Unreleased

- Split the original local pipeline into `backend/` and `frontend/` projects while preserving the PDF -> GROBID/LightOCR -> JSON-LD ModelCard flow
- Added Poetry environments, Dockerfiles, `.dockerignore` files, and Docker Compose services for Ollama/model init, GROBID, RabbitMQ, backend, worker, and frontend
- Added FastAPI job endpoints: `POST /job/launch-job` and `GET /job/job-status/{job_id}` with persisted `status.json` files under `DATA_DIR/jobs`.
- Added RabbitMQ job publishing and a worker with dummy mode for smoke tests and real `PDFHandler` mode for full pipeline runs
- Moved backend code to package-style imports and environment-driven config, including shared `DATA_DIR` support for Docker bind mounts.
- Hardened the real worker path with 600s external timeouts, explicit PDF/GROBID/OCR failures, corrected dataset extraction inputs, and worker-visible `PDFHandler` progress logs
- Added backend/frontend `.dockerignore` files and kept only a small runtime data sample

### Frontend/backend job integration

- Added a Streamlit API client configured by `P2MC_API_URL` to submit arXiv URLs to `POST /job/launch-job`.
- Connected the main Streamlit form to FastAPI job launch and display the queued job ID, arXiv ID, and initial status.
- Added manual job status refresh in Streamlit, including status metadata, errors, artifacts, and downloadable JSON-LD ModelCard output when completed.
- Added `GET /job/jobs` and safe artifact downloads through `GET /job/{job_id}/artifacts/{artifact_name}`, plus a Streamlit Jobs page to browse previous file-backed jobs.
- Configured the frontend Compose service to call the backend through the Docker network.

### Memory usage relief

- Reduced worker peak memory by keeping `PDFHandler` initialized with `SciPdfParser` only, instead of holding LightOCR and ModelCard generation models for the whole worker lifetime.
- Changed LightOCR processing to load `LightOcrParser` only during OCR, skip loading it when the OCR JSON already exists, and release it before ModelCard generation starts.
- Changed ModelCard generation to instantiate `ModelCardGenerator` only after PDF/XML/OCR artifacts have been extracted, then release it after the ModelCard is produced.
- Stopped active initialization of unused ModelCard extractors while leaving their imports and constructor lines commented for future reactivation.
- Deferred the `transformers` import used by `QwenExtractor` so importing the shared LLM extractor module does not load Transformers unless that extractor is instantiated.
