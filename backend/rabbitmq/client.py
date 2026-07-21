import json
import socket
import time

import pika

from backend import (
    QUEUE_NAME,
    RABBITMQ_BLOCKED_CONNECTION_TIMEOUT_SECONDS,
    RABBITMQ_HEARTBEAT_SECONDS,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_RETRY_DELAY_SECONDS,
    RABBITMQ_USER,
)


RABBITMQ_PUBLISH_MAX_ATTEMPTS = 3
RABBITMQ_CONNECTION_ERRORS = (
    pika.exceptions.AMQPConnectionError,
    pika.exceptions.AMQPError,
    socket.gaierror,
)


class RabbitMQPublishError(RuntimeError):
    """Error al publicar un trabajo en RabbitMQ."""


def _open_connection():
    credentials = pika.PlainCredentials(
        RABBITMQ_USER,
        RABBITMQ_PASSWORD,
    )

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=RABBITMQ_HEARTBEAT_SECONDS,
            blocked_connection_timeout=(
                RABBITMQ_BLOCKED_CONNECTION_TIMEOUT_SECONDS
            ),
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    return connection, channel


def rabbit_connect():
    while True:
        try:
            connection, channel = _open_connection()
            print("RabbitMQ conexion set")
            return connection, channel

        except RABBITMQ_CONNECTION_ERRORS:
            print(
                "RabbitMQ not ready, retrying in "
                f"{RABBITMQ_RETRY_DELAY_SECONDS}s...",
                flush=True,
            )
            time.sleep(RABBITMQ_RETRY_DELAY_SECONDS)


def rabbit_connect_for_publish(
    max_attempts: int = RABBITMQ_PUBLISH_MAX_ATTEMPTS,
):
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return _open_connection()

        except RABBITMQ_CONNECTION_ERRORS as exc:
            last_error = exc
            if attempt < max_attempts:
                print(
                    "RabbitMQ not ready for publish "
                    f"({attempt}/{max_attempts}), retrying in "
                    f"{RABBITMQ_RETRY_DELAY_SECONDS}s...",
                    flush=True,
                )
                time.sleep(RABBITMQ_RETRY_DELAY_SECONDS)

    raise RabbitMQPublishError(
        "RabbitMQ is not available for publishing"
    ) from last_error


def publish_job(url: str, job_id: str, arxiv_id: str) -> None:
    connection = None

    message = {
        "url": url,
        "job_id": job_id,
        "arxiv_id": arxiv_id,
    }

    try:
        connection, channel = rabbit_connect_for_publish()

        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
            mandatory=True,
        )

    except RABBITMQ_CONNECTION_ERRORS as exc:
        raise RabbitMQPublishError(
            f"No se pudo publicar el job {job_id}"
        ) from exc

    except RabbitMQPublishError as exc:
        raise RabbitMQPublishError(
            f"No se pudo publicar el job {job_id}"
        ) from exc

    finally:
        if connection is not None and connection.is_open:
            connection.close()
