from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pika

from backend import (
    DATA_DIR,
    QUEUE_NAME,
    RABBITMQ_RETRY_DELAY_SECONDS,
)
from backend.rabbitmq.client import rabbit_connect
from backend.pdf_handler import PDFHandler


ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")
DATA_ROOT = Path(DATA_DIR).resolve()
USE_DUMMY_WORKER = os.getenv(
    "P2MC_USE_DUMMY_WORKER",
    "true",
).lower() in {"1", "true", "yes"}
PIPELINE_TOTAL_STEPS = 7


def timestamp(message: str) -> None:
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pipeline_stage(
    key: str,
    label: str,
    step: int | None = None,
    *,
    detail: str | None = None,
    item_current: int | None = None,
    item_total: int | None = None,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "key": key,
        "label": label,
    }

    if step is not None:
        stage["step"] = step
        stage["total"] = PIPELINE_TOTAL_STEPS

    if detail is not None:
        stage["detail"] = detail

    if item_current is not None:
        stage["item_current"] = item_current

    if item_total is not None:
        stage["item_total"] = item_total

    return stage


def get_status_path(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id / "status.json"


def read_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.is_file():
        return None

    with status_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid status JSON: {status_path}")

    return data


def write_status_atomic(
    status_path: Path,
    status_data: dict[str, Any],
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = status_path.with_name(
        f".{status_path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                status_data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, status_path)

    finally:
        temporary_path.unlink(missing_ok=True)


def validate_message(
    message: dict[str, Any],
) -> tuple[str, str, str]:
    job_id = str(message["job_id"])
    arxiv_id = str(message["arxiv_id"])
    url = str(message["url"])

    # Valida y normaliza el UUID.
    job_id = str(UUID(job_id))

    if ARXIV_ID_PATTERN.fullmatch(arxiv_id) is None:
        raise ValueError(
            f"Invalid base arXiv ID: {arxiv_id}"
        )

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}")

    return job_id, arxiv_id, url


def get_artifact_paths(arxiv_id: str) -> dict[str, Path]:
    return {
        "pdf": DATA_ROOT / "raw" / "pdfs" / f"{arxiv_id}.pdf",
        "xml": (
            DATA_ROOT
            / "interim"
            / "scipdf_xml"
            / f"{arxiv_id}.xml"
        ),
        "lightocr_json": (
            DATA_ROOT
            / "interim"
            / "lightocr_json"
            / f"{arxiv_id}.json"
        ),
        "modelcard": (
            DATA_ROOT
            / "processed"
            / "modelcards"
            / f"{arxiv_id}_modelcard.json"
        ),
    }


def serialize_artifact_paths(
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    return {
        name: path.relative_to(DATA_ROOT).as_posix()
        for name, path in artifact_paths.items()
    }


def serialize_existing_artifact_paths(
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    return {
        name: path.relative_to(DATA_ROOT).as_posix()
        for name, path in artifact_paths.items()
        if path.is_file()
    }


def read_existing_modelcard(
    artifact_paths: dict[str, Path],
) -> dict[str, Any] | None:
    modelcard_path = artifact_paths["modelcard"]

    if not modelcard_path.is_file():
        return None

    try:
        with modelcard_path.open("r", encoding="utf-8") as file:
            card = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(card, dict):
        return None

    return card


def verify_artifacts(
    artifact_paths: dict[str, Path],
) -> None:
    missing = [
        str(path)
        for path in artifact_paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "P2MC finished without generating all expected files: "
            + ", ".join(missing)
        )


def process_job(message: dict[str, Any], pdf_handler: PDFHandler) -> None:
    job_id, arxiv_id, url = validate_message(message)
    status_path = get_status_path(job_id)
    previous_status = read_status(status_path) or {}

    # RabbitMQ puede reenviar el mensaje si el worker terminó pero cayó
    # antes del ACK. No repetimos un job que ya alcanzó un estado terminal.
    if previous_status.get("status") in {"completed", "failed"}:
        timestamp(
            f"Job {job_id} already finished as "
            f"{previous_status['status']}; skipping"
        )
        return

    created_at = previous_status.get("created_at") or utc_now()
    started_at = utc_now()

    processing_status = {
        **previous_status,
        "job_id": job_id,
        "arxiv_id": arxiv_id,
        "url": url,
        "status": "processing",
        "created_at": created_at,
        "started_at": started_at,
        "updated_at": started_at,
        "completed_at": None,
        "error": None,
        "artifacts": None,
        "card": None,
        "pipeline_stage": build_pipeline_stage(
            "initializing",
            "Preparing paper processing",
            1,
        ),
    }
    write_status_atomic(status_path, processing_status)

    timestamp(f"Processing job {job_id} for arXiv {arxiv_id}")

    try:
        artifact_paths = get_artifact_paths(arxiv_id)
        cached_card = read_existing_modelcard(artifact_paths)

        if cached_card is not None:
            completed_at = utc_now()
            completed_status = {
                **processing_status,
                "status": "completed",
                "updated_at": completed_at,
                "completed_at": completed_at,
                "error": None,
                "artifacts": serialize_existing_artifact_paths(
                    artifact_paths
                ),
                "card": cached_card,
                "pipeline_stage": build_pipeline_stage(
                    "completed",
                    "Using existing ModelCard",
                    PIPELINE_TOTAL_STEPS,
                    detail="Paper already processed.",
                ),
            }
            write_status_atomic(status_path, completed_status)
            timestamp(
                f"Job {job_id} reused existing ModelCard for {arxiv_id}"
            )
            return

        def update_stage(stage: dict[str, Any]) -> None:
            current_status = read_status(status_path) or processing_status
            write_status_atomic(
                status_path,
                {
                    **current_status,
                    "status": "processing",
                    "updated_at": utc_now(),
                    "pipeline_stage": stage,
                },
            )

        card = pdf_handler.handle_pdf(url, on_stage=update_stage)

        if not isinstance(card, dict):
            raise RuntimeError(
                "PDFHandler returned a value that is not a dictionary"
            )

        # PDFHandler captura internamente sus excepciones y devuelve
        # un diccionario con la clave error.
        if card.get("error"):
            raise RuntimeError(str(card["error"]))

        verify_artifacts(artifact_paths)

    except Exception as exc:
        failed_at = utc_now()
        current_status = read_status(status_path) or processing_status
        previous_stage = current_status.get("pipeline_stage")
        failed_step = PIPELINE_TOTAL_STEPS
        failed_detail = None

        if isinstance(previous_stage, dict):
            failed_step = previous_stage.get("step") or failed_step
            failed_detail = (
                previous_stage.get("detail")
                or previous_stage.get("label")
            )

        failed_status = {
            **current_status,
            "status": "failed",
            "updated_at": failed_at,
            "completed_at": failed_at,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "pipeline_stage": build_pipeline_stage(
                "failed",
                "Failed",
                failed_step,
                detail=failed_detail,
            ),
        }
        write_status_atomic(status_path, failed_status)

        timestamp(f"Job {job_id} failed: {exc}")
        traceback.print_exc()
        return

    completed_at = utc_now()
    current_status = read_status(status_path) or processing_status
    completed_status = {
        **current_status,
        "status": "completed",
        "updated_at": completed_at,
        "completed_at": completed_at,
        "error": None,
        "artifacts": serialize_artifact_paths(artifact_paths),
        "card": card,
        "pipeline_stage": build_pipeline_stage(
            "completed",
            "Completed",
            PIPELINE_TOTAL_STEPS,
        ),
    }
    write_status_atomic(status_path, completed_status)

    timestamp(f"Job {job_id} completed")


def process_message(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
    pdf_handler: PDFHandler | None = None,
) -> None:
    del properties

    try:
        decoded_message = json.loads(body.decode("utf-8"))

        if not isinstance(decoded_message, dict):
            raise ValueError("RabbitMQ message must contain a JSON object")

        if USE_DUMMY_WORKER:
            process_job_dummy(decoded_message)
        else:
            if pdf_handler is None:
                raise RuntimeError(
                    "PDFHandler is required when dummy worker is disabled"
                )
            process_job(decoded_message, pdf_handler)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        # Mensaje inválido: volver a ponerlo en la cola causaría un bucle.
        timestamp(f"Rejecting invalid message: {exc}")
        channel.basic_reject(
            delivery_tag=method.delivery_tag,
            requeue=False,
        )

    except Exception as exc:
        # Fallo de infraestructura, por ejemplo al escribir status.json.
        # Se reencola porque puede ser temporal.
        timestamp(f"Worker infrastructure error: {exc}")
        traceback.print_exc()
        channel.basic_nack(
            delivery_tag=method.delivery_tag,
            requeue=True,
        )

    else:
        channel.basic_ack(delivery_tag=method.delivery_tag)


def worker() -> None:
    timestamp("P2MC worker started")
    pdf_handler = None

    if not USE_DUMMY_WORKER:
        timestamp("PDFHandler: initialization started")
        pdf_handler = PDFHandler(logger=timestamp)
        timestamp("PDFHandler: initialization finished")

    def callback(channel, method, properties, body):
        process_message(
            channel,
            method,
            properties,
            body,
            pdf_handler,
            )
    while True:
        connection: pika.BlockingConnection | None = None

        try:
            connection, channel = rabbit_connect()

            channel.queue_declare(
                queue=QUEUE_NAME,
                durable=True,
            )
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue=QUEUE_NAME,
                on_message_callback=callback,
                auto_ack=False,
            )

            timestamp(f"Waiting for jobs in {QUEUE_NAME}")
            channel.start_consuming()

        except KeyboardInterrupt:
            timestamp("Stopping P2MC worker")
            break

        except pika.exceptions.AMQPError as exc:
            timestamp(
                f"RabbitMQ connection lost: {exc}. "
                f"Retrying in {RABBITMQ_RETRY_DELAY_SECONDS}s"
            )
            time.sleep(RABBITMQ_RETRY_DELAY_SECONDS)

        finally:
            if connection is not None and connection.is_open:
                connection.close()









def process_job_dummy(message: dict[str, Any]) -> None:
    job_id, arxiv_id, url = validate_message(message)

    status_path = get_status_path(job_id)
    previous_status = read_status(status_path) or {}

    if previous_status.get("status") in {"completed", "failed"}:
        timestamp(
            f"Dummy job {job_id} already finished as "
            f"{previous_status['status']}; skipping"
        )
        return

    created_at = previous_status.get("created_at") or utc_now()
    started_at = utc_now()

    processing_status = {
        **previous_status,
        "job_id": job_id,
        "arxiv_id": arxiv_id,
        "url": url,
        "status": "processing",
        "created_at": created_at,
        "started_at": started_at,
        "updated_at": started_at,
        "completed_at": None,
        "error": None,
        "artifacts": None,
        "card": None,
        "pipeline_stage": build_pipeline_stage(
            "processing_dummy",
            "Running dummy processor",
            1,
        ),
    }

    write_status_atomic(
        status_path,
        processing_status,
    )

    timestamp(
        f"Processing dummy job {job_id} "
        f"for arXiv {arxiv_id}"
    )

    try:
        # Simula que P2MC tarda en procesar el paper.
        time.sleep(10)

        artifact_paths = get_artifact_paths(arxiv_id)

        # Creamos los directorios necesarios.
        for artifact_path in artifact_paths.values():
            artifact_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        # PDF ficticio. No es un PDF real, solo sirve para comprobar
        # que el worker crea el archivo y registra su ruta.
        if not artifact_paths["pdf"].is_file():
            artifact_paths["pdf"].write_bytes(
                b"%PDF-1.4\n"
                b"% Dummy P2MC PDF\n"
                b"%%EOF\n"
            )

        artifact_paths["xml"].write_text(
            (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                "<article>\n"
                f"  <id>{arxiv_id}</id>\n"
                "  <title>Dummy P2MC paper</title>\n"
                "</article>\n"
            ),
            encoding="utf-8",
        )

        lightocr_data = {
            "arxiv_id": arxiv_id,
            "tables": [
                {
                    "caption": "Dummy table",
                    "rows": [
                        ["Metric", "Value"],
                        ["accuracy", "0.95"],
                    ],
                }
            ],
        }

        with artifact_paths["lightocr_json"].open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                lightocr_data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        card = {
            "@context": "https://schema.org",
            "@type": "CreativeWork",
            "name": "Dummy P2MC ModelCard",
            "identifier": arxiv_id,
            "url": url,
            "description": (
                "Dummy ModelCard generated to test "
                "the RabbitMQ worker."
            ),
            "status": "test",
        }

        with artifact_paths["modelcard"].open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                card,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        verify_artifacts(artifact_paths)

    except Exception as exc:
        failed_at = utc_now()

        failed_status = {
            **processing_status,
            "status": "failed",
            "updated_at": failed_at,
            "completed_at": failed_at,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "pipeline_stage": build_pipeline_stage(
                "failed",
                "Failed",
                1,
                detail="Running dummy processor",
            ),
        }

        write_status_atomic(
            status_path,
            failed_status,
        )

        timestamp(
            f"Dummy job {job_id} failed: {exc}"
        )
        traceback.print_exc()
        return

    completed_at = utc_now()

    completed_status = {
        **processing_status,
        "status": "completed",
        "updated_at": completed_at,
        "completed_at": completed_at,
        "error": None,
        "artifacts": serialize_artifact_paths(
            artifact_paths
        ),
        "card": card,
        "pipeline_stage": build_pipeline_stage(
            "completed",
            "Completed",
            PIPELINE_TOTAL_STEPS,
        ),
    }

    write_status_atomic(
        status_path,
        completed_status,
    )

    timestamp(
        f"Dummy job {job_id} completed"
    )
    
    
    
   
   
   
   
   
    
    
    

if __name__ == "__main__":
    worker()
