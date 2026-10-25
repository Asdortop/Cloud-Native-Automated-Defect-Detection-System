"""
src/serving/consumer.py
------------------------
Kafka consumer that listens to the ``new-images`` topic, runs ONNX inference
on each image, and writes the prediction result to a Postgres ``predictions``
table.  Simulates the "queue → inference → results-DB" leg of the pipeline.

Architecture
------------

  Camera  →  [new-images topic]  →  InferenceConsumer  →  [Postgres]
                                           │
                                           └─► [inference-results topic]  (fan-out)
                                           └─► [inference-dlq topic]       (on error)

Delivery guarantees
-------------------
  - ``enable_auto_commit=False`` — offsets are committed ONLY after the
    prediction has been successfully written to Postgres.
  - A message that fails repeatedly is forwarded to the DLQ topic instead
    of blocking the partition forever.
  - At-least-once semantics (rare duplicate on crash between DB write and
    offset commit); the Postgres insert is idempotent on ``job_id``.

Configuration (environment variables)
--------------------------------------
  KAFKA_BOOTSTRAP_SERVERS   Broker addresses           (default: localhost:9092)
  KAFKA_TOPIC_IMAGES        Source topic               (default: new-images)
  KAFKA_TOPIC_RESULTS       Results fan-out topic      (default: inference-results)
  KAFKA_TOPIC_DLQ           Dead-letter topic          (default: inference-dlq)
  KAFKA_GROUP_ID            Consumer group ID          (default: inference-worker)
  KAFKA_AUTO_OFFSET_RESET   earliest | latest          (default: latest)
  ONNX_MODEL_PATH           Path to .onnx model        (required)
  CLASS_NAMES               Comma-separated classes    (required)
  POSTGRES_DSN              PostgreSQL connection DSN   (required)
  MINIO_ENDPOINT            MinIO URL for s3:// paths  (default: http://localhost:9000)
  MINIO_ACCESS_KEY          MinIO access key           (default: minioadmin)
  MINIO_SECRET_KEY          MinIO secret key           (default: minioadmin)
  MAX_RETRIES               Retries before DLQ         (default: 3)

Running
-------
  # Standalone
  python -m src.serving.consumer

  # In Docker (command override in docker-compose)
  command: ["python", "-m", "src.serving.consumer"]

  # Scale out — each pod joins the same consumer group, partitions are
  # distributed automatically:
  docker compose up --scale inference-worker=3
"""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from PIL import Image

# Shared preprocessing — identical to what main.py and training use
from src.data.transforms import get_transforms

# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(job_id)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# A log adapter that injects job_id into every record automatically
_EXTRA_DEFAULTS: dict[str, str] = {"job_id": "-"}


def _logger(job_id: str = "-") -> logging.LoggerAdapter:
    return logging.LoggerAdapter(log, {"job_id": job_id})


# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS:      str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_IMAGES:           str = os.getenv("KAFKA_TOPIC_IMAGES",      "new-images")
TOPIC_RESULTS:          str = os.getenv("KAFKA_TOPIC_RESULTS",     "inference-results")
TOPIC_DLQ:              str = os.getenv("KAFKA_TOPIC_DLQ",         "inference-dlq")
GROUP_ID:               str = os.getenv("KAFKA_GROUP_ID",          "inference-worker")
AUTO_OFFSET_RESET:      str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")
ONNX_MODEL_PATH:        str = os.getenv("ONNX_MODEL_PATH",         "models/defect_efficientnet_b0.onnx")
CLASS_NAMES_RAW:        str = os.getenv("CLASS_NAMES",             "Crazing,Inclusion,Patches,Pitted_surface,Rolled-in_scale,Scratches")
CLASS_NAMES:            list[str] = [c.strip() for c in CLASS_NAMES_RAW.split(",") if c.strip()]
POSTGRES_DSN:           str = os.getenv("POSTGRES_DSN",            "postgresql://mlflow:mlflow@localhost:5432/mlflow")
MINIO_ENDPOINT:         str = os.getenv("MINIO_ENDPOINT",          "http://localhost:9000")
MINIO_ACCESS_KEY:       str = os.getenv("MINIO_ACCESS_KEY",        "minioadmin")
MINIO_SECRET_KEY:       str = os.getenv("MINIO_SECRET_KEY",        "minioadmin")
MAX_RETRIES:            int = int(os.getenv("MAX_RETRIES",          "3"))
CONNECT_RETRY_SECONDS:  int = 30
CONNECT_RETRY_INTERVAL: float = 2.0


# ---------------------------------------------------------------------------
# Postgres DDL
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    job_id           UUID        NOT NULL,
    camera_id        VARCHAR(64) NOT NULL DEFAULT '',
    image_path       TEXT        NOT NULL,
    predicted_class  VARCHAR(64) NOT NULL,
    class_index      INT         NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL,
    probabilities    JSONB       NOT NULL DEFAULT '{}',
    inference_ms     DOUBLE PRECISION,
    captured_at      TIMESTAMPTZ,
    consumed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error            TEXT,                -- NULL on success
    CONSTRAINT predictions_job_id_uq UNIQUE (job_id)   -- idempotent inserts
);

CREATE INDEX IF NOT EXISTS predictions_camera_idx ON predictions (camera_id);
CREATE INDEX IF NOT EXISTS predictions_class_idx  ON predictions (predicted_class);
CREATE INDEX IF NOT EXISTS predictions_time_idx   ON predictions (consumed_at DESC);
"""


# ===========================================================================
# Image loading (local filesystem + MinIO s3://)
# ===========================================================================

def _load_image_bytes(image_path: str) -> bytes:
    """
    Load raw image bytes from a local path or an ``s3://bucket/key`` URI.

    For s3:// paths the MinIO client is used (boto3 under the hood).
    """
    if image_path.startswith("s3://"):
        return _load_from_minio(image_path)
    return Path(image_path).read_bytes()


def _load_from_minio(s3_uri: str) -> bytes:
    """Download an object from MinIO given an ``s3://bucket/key`` URI."""
    try:
        import boto3
        from botocore.client import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for s3:// image paths. "
            "Install it with: pip install boto3"
        ) from exc

    # Parse s3://bucket/key/…
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


# ===========================================================================
# Inference helpers
# ===========================================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D float array."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _run_inference(
    session: ort.InferenceSession,
    input_name: str,
    transform,
    raw_bytes: bytes,
    class_names: list[str],
) -> dict[str, Any]:
    """
    End-to-end inference: bytes → PIL → transform → ORT → softmax.

    Returns a dict ready to INSERT into the predictions table.
    """
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    tensor    = transform(pil_image)                       # (3, 224, 224)
    input_np  = tensor.unsqueeze(0).numpy()                # (1, 3, 224, 224)

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: input_np})
    inference_ms = (time.perf_counter() - t0) * 1000

    logits: np.ndarray = outputs[0][0]
    probs              = _softmax(logits)
    pred_idx: int      = int(np.argmax(probs))
    confidence: float  = float(probs[pred_idx])
    pred_class: str    = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

    return {
        "predicted_class": pred_class,
        "class_index":     pred_idx,
        "confidence":      confidence,
        "probabilities":   {
            (class_names[i] if i < len(class_names) else f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        },
        "inference_ms": round(inference_ms, 2),
    }


# ===========================================================================
# InferenceConsumer
# ===========================================================================

class InferenceConsumer:
    """
    Long-running daemon that:
    1. Polls ``new-images`` topic for image-reference events.
    2. Loads each image (local fs or MinIO).
    3. Runs ONNX inference.
    4. Writes prediction to Postgres.
    5. Fan-outs result to ``inference-results`` topic.
    6. Commits Kafka offset (at-least-once guarantee).
    7. On repeated failure, forwards to DLQ and moves on.
    """

    def __init__(self) -> None:
        self._running = False
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.transform = None
        self.pg_conn: Optional[psycopg2.extensions.connection] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.result_producer: Optional[KafkaProducer] = None

    # ── Startup ───────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        log.info("Loading ONNX model from: %s", ONNX_MODEL_PATH, extra=_EXTRA_DEFAULTS)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.transform  = get_transforms(train=False)
        log.info("Model ready. input='%s'", self.input_name, extra=_EXTRA_DEFAULTS)

    def _init_postgres(self) -> None:
        log.info("Connecting to Postgres: %s", POSTGRES_DSN, extra=_EXTRA_DEFAULTS)
        self.pg_conn = psycopg2.connect(POSTGRES_DSN)
        self.pg_conn.autocommit = False
        with self.pg_conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
        self.pg_conn.commit()
        log.info("Postgres ready. predictions table ensured.", extra=_EXTRA_DEFAULTS)

    def _init_kafka(self) -> None:
        servers = [s.strip() for s in BOOTSTRAP_SERVERS.split(",")]
        log.info("Connecting to Kafka brokers: %s", servers, extra=_EXTRA_DEFAULTS)

        deadline = time.monotonic() + CONNECT_RETRY_SECONDS
        while True:
            try:
                self.consumer = KafkaConsumer(
                    TOPIC_IMAGES,
                    bootstrap_servers=servers,
                    group_id=GROUP_ID,
                    client_id=f"inference-worker-{os.getpid()}",
                    enable_auto_commit=False,   # manual commit after DB write
                    auto_offset_reset=AUTO_OFFSET_RESET,
                    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                    # Fetch up to 10 messages per poll for micro-batching
                    max_poll_records=10,
                    session_timeout_ms=30_000,
                    heartbeat_interval_ms=10_000,
                )
                self.result_producer = KafkaProducer(
                    bootstrap_servers=servers,
                    client_id="inference-result-producer",
                    acks=1,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    compression_type="gzip",
                )
                log.info(
                    "Kafka ready. group=%s topic=%s",
                    GROUP_ID, TOPIC_IMAGES,
                    extra=_EXTRA_DEFAULTS,
                )
                return
            except NoBrokersAvailable:
                if time.monotonic() >= deadline:
                    raise
                log.warning(
                    "Kafka not yet available — retrying in %.1fs …",
                    CONNECT_RETRY_INTERVAL,
                    extra=_EXTRA_DEFAULTS,
                )
                time.sleep(CONNECT_RETRY_INTERVAL)

    def start(self) -> None:
        """Initialise all resources and start the poll loop."""
        self._init_model()
        self._init_postgres()
        self._init_kafka()
        self._running = True
        self._run_loop()

    # ── Main poll loop ─────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        log.info("Consumer started. Waiting for messages …", extra=_EXTRA_DEFAULTS)
        retry_counts: dict[str, int] = {}   # job_id → retry count

        try:
            while self._running:
                records = self.consumer.poll(timeout_ms=1000)  # 1-second poll
                if not records:
                    continue

                for _tp, messages in records.items():
                    for msg in messages:
                        self._handle_message(msg, retry_counts)

        except KeyboardInterrupt:
            log.info("Interrupted — shutting down.", extra=_EXTRA_DEFAULTS)
        finally:
            self._shutdown()

    def _handle_message(
        self,
        msg: Any,
        retry_counts: dict[str, int],
    ) -> None:
        """
        Process one Kafka message — infer, write to DB, commit offset.
        On unrecoverable error, route to DLQ.
        """
        event: dict = msg.value
        job_id: str = event.get("job_id", "unknown")
        lg = _logger(job_id)

        lg.info(
            "Received offset=%d partition=%d camera=%s path=%s",
            msg.offset, msg.partition,
            event.get("camera_id"), event.get("image_path"),
        )

        image_path  = event.get("image_path", "")
        camera_id   = event.get("camera_id", "")
        captured_at = event.get("timestamp")
        metadata    = event.get("metadata", {})

        try:
            # ── 1. Load image ─────────────────────────────────────────────
            raw_bytes = _load_image_bytes(image_path)

            # ── 2. Infer ──────────────────────────────────────────────────
            result = _run_inference(
                self.session, self.input_name,
                self.transform, raw_bytes, CLASS_NAMES,
            )
            lg.info(
                "Predicted class='%s' conf=%.4f in %.1f ms",
                result["predicted_class"], result["confidence"], result["inference_ms"],
            )

            # ── 3. Write to Postgres ──────────────────────────────────────
            self._write_prediction(
                job_id=job_id,
                camera_id=camera_id,
                image_path=image_path,
                captured_at=captured_at,
                metadata=metadata,
                result=result,
            )

            # ── 4. Fan-out to inference-results topic ─────────────────────
            self._publish_result(event, result)

            # ── 5. Commit offset (success path) ───────────────────────────
            self.consumer.commit()
            retry_counts.pop(job_id, None)

        except Exception as exc:
            lg.error("Error processing message: %s", exc, exc_info=True)
            retries = retry_counts.get(job_id, 0) + 1
            retry_counts[job_id] = retries

            if retries >= MAX_RETRIES:
                lg.warning(
                    "Max retries (%d) reached — routing to DLQ.", MAX_RETRIES,
                )
                self._publish_dlq(event, str(exc))
                # Commit to advance past the poison pill
                self.consumer.commit()
                retry_counts.pop(job_id, None)
            else:
                lg.info("Will retry (attempt %d/%d).", retries, MAX_RETRIES)
                # Do NOT commit — the message will be re-delivered on next poll

    # ── Database write ────────────────────────────────────────────────────

    def _write_prediction(
        self,
        job_id: str,
        camera_id: str,
        image_path: str,
        captured_at: Optional[str],
        metadata: dict,
        result: dict[str, Any],
    ) -> None:
        """
        Insert a prediction row.  ON CONFLICT DO NOTHING provides idempotency
        so duplicate deliveries (after crash) are harmless.
        """
        sql = """
            INSERT INTO predictions
                (job_id, camera_id, image_path, predicted_class, class_index,
                 confidence, probabilities, inference_ms, captured_at)
            VALUES
                (%(job_id)s, %(camera_id)s, %(image_path)s, %(predicted_class)s,
                 %(class_index)s, %(confidence)s, %(probabilities)s,
                 %(inference_ms)s, %(captured_at)s)
            ON CONFLICT (job_id) DO NOTHING
        """
        params = {
            "job_id":          job_id,
            "camera_id":       camera_id,
            "image_path":      image_path,
            "predicted_class": result["predicted_class"],
            "class_index":     result["class_index"],
            "confidence":      result["confidence"],
            "probabilities":   psycopg2.extras.Json(result["probabilities"]),
            "inference_ms":    result["inference_ms"],
            "captured_at":     captured_at,
        }
        with self.pg_conn.cursor() as cur:
            cur.execute(sql, params)
        self.pg_conn.commit()

    # ── Kafka fan-outs ────────────────────────────────────────────────────

    def _publish_result(self, event: dict, result: dict) -> None:
        """Publish enriched result to the inference-results topic."""
        if self.result_producer is None:
            return
        payload = {**event, **result, "consumed_at": datetime.now(timezone.utc).isoformat()}
        try:
            self.result_producer.send(
                TOPIC_RESULTS,
                key=(event.get("camera_id", "") or "").encode(),
                value=json.dumps(payload).encode(),
            )
        except KafkaError as exc:
            log.warning("Could not publish to results topic: %s", exc, extra=_EXTRA_DEFAULTS)

    def _publish_dlq(self, event: dict, error: str) -> None:
        """Route a poison-pill message to the dead-letter queue topic."""
        if self.result_producer is None:
            return
        payload = {
            **event,
            "dlq_reason": error,
            "dlq_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.result_producer.send(
                TOPIC_DLQ,
                value=json.dumps(payload).encode(),
            )
            self.result_producer.flush(timeout=5)
        except KafkaError as exc:
            log.error("Could not publish to DLQ: %s", exc, extra=_EXTRA_DEFAULTS)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the poll loop to exit gracefully."""
        log.info("Stop requested.", extra=_EXTRA_DEFAULTS)
        self._running = False

    def _shutdown(self) -> None:
        """Close all resources cleanly."""
        if self.consumer:
            self.consumer.close()
        if self.result_producer:
            self.result_producer.flush(timeout=5)
            self.result_producer.close()
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
        log.info("Consumer shut down cleanly.", extra=_EXTRA_DEFAULTS)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    worker = InferenceConsumer()

    # Graceful shutdown on SIGTERM (Docker stop) and SIGINT (Ctrl-C)
    def _sig_handler(signum: int, _frame: Any) -> None:
        log.info("Signal %s received — stopping.", signal.Signals(signum).name,
                 extra=_EXTRA_DEFAULTS)
        worker.stop()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT,  _sig_handler)

    try:
        worker.start()
    except Exception as exc:
        log.critical("Worker crashed: %s", exc, exc_info=True, extra=_EXTRA_DEFAULTS)
        sys.exit(1)


if __name__ == "__main__":
    main()
