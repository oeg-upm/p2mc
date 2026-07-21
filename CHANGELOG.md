# Changelog

## General

- Split the original local pipeline into `backend/` and `frontend/` projects while preserving the PDF -> GROBID/LightOCR -> JSON-LD ModelCard flow
- Added Poetry environments, Dockerfiles, `.dockerignore` files, and Docker Compose services for Ollama/model init, GROBID, RabbitMQ, backend, worker, and frontend
- Added FastAPI job endpoints: `POST /job/launch-job` and `GET /job/job-status/{job_id}` with persisted `status.json` files under `DATA_DIR/jobs`.
- Added RabbitMQ job publishing and a worker with dummy mode for smoke tests and real `PDFHandler` mode for full pipeline runs
- Moved backend code to package-style imports and environment-driven config, including shared `DATA_DIR` support for Docker bind mounts.
- Hardened the real worker path with 600s external timeouts, explicit PDF/GROBID/OCR failures, corrected dataset extraction inputs, and worker-visible `PDFHandler` progress logs
- Made LPWC dataset URI checks degrade to generated fallback URIs when SPARQL/network/SSL requests fail.
- Added backend/frontend `.dockerignore` files and kept only a small runtime data sample
- Updated README to describe the Docker-first workflow and current project structure

### Job progress and cached results

- Added `pipeline_stage` to job status responses so the frontend can show the active pipeline step and LightOCR page progress.
- Reused existing generated ModelCards for already processed arXiv papers instead of enqueueing/rerunning the full pipeline.
- Treated invalid cached ModelCard JSON as a cache miss so the worker can regenerate the paper instead of failing immediately.
- Changed the frontend job panel to auto-refresh active jobs and show the returned JSON-LD ModelCard inline for copying.

### Frontend/backend job integration

- Added a Streamlit API client configured by `P2MC_API_URL` to submit arXiv URLs to `POST /job/launch-job`
- Connected the main Streamlit form to FastAPI job launch and display the queued job ID, arXiv ID, and initial status
- Added manual job status refresh in Streamlit, including status metadata, errors, and generated XML/OCR previews when completed
- Added `GET /job/jobs` and viewable XML/OCR content through `GET /job/{job_id}/artifacts/{artifact_name}`, plus a Streamlit Jobs page to browse previous file-backed jobs
- Configured the frontend Compose service to call the backend through the Docker network

### Memory usage relief

- Reduced worker peak memory by keeping `PDFHandler` initialized with `SciPdfParser` only, instead of holding LightOCR and ModelCard generation models for the whole worker lifetime
- Changed LightOCR processing to load `LightOcrParser` only during OCR, skip loading it when the OCR JSON already exists, and release it before ModelCard generation starts
- Reduced LightOCR's default image budget from `1540` to `1024` and token budget from `8192` to `1024` to avoid Docker OOM kills during local real-worker runs
- Added `P2MC_LIGHTOCR_TARGET_LONGEST`, `P2MC_LIGHTOCR_MAX_NEW_TOKENS`, and `P2MC_LIGHTOCR_MODEL_ID` so LightOCR quality/memory can be tuned per machine without rebuilding the image
- Logged the active LightOCR model and budgets at startup so worker logs show which memory profile is actually running
- Routed LightOCR page progress through the worker logger so `docker compose logs -f p2mc-worker` shows render/OCR start, finish, detected table counts, and total tables per PDF
- Loaded LightOCR with `low_cpu_mem_usage=True`, switched it to `eval()`/`torch.inference_mode()`, and explicitly released per-page tensors/images to reduce inference-time memory spikes.
- Changed ModelCard generation to instantiate `ModelCardGenerator` only after PDF/XML/OCR artifacts have been extracted, then release it after the ModelCard is produced.
- Stopped active initialization of unused ModelCard extractors while leaving their imports and constructor lines commented for future reactivation
- Deferred the `transformers` import used by `QwenExtractor` so importing the shared LLM extractor module does not load Transformers unless that extractor is instantiated


## Issues fixed
https://github.com/oeg-upm/p2mc/issues/4
