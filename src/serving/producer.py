"""
src/serving/producer.py
------------------------
Publishes image-reference events to the Redpanda (Kafka-API) ``new-images``
topic so downstream inference workers can pick them up asynchronously.

This module simulates the "camera → queue" leg of the pipeline.  A real
camera controller would call ``publish_image()`` after each frame capture;
the image itself lives in MinIO (S3) or on a shared NFS mount.  Only its
reference (path / URL) travels over Kafka, keeping message sizes small.

Message schema (JSON)
---------------------
{
  "job_id":     "3f4a8c1d-...",   # UUID4 — unique per capture event
  "camera_id":  "cam-line-01",    # logical camera / sensor identifier
  "image_path": "s3://camera-images/2024/line-01/img_001.bmp",
                                  # OR a local path for dev/testing
  "timestamp":  "2024-01-15T08:23:41.123456Z",
  "metadata":   { "line": "A", "shift": "morning" }   # optional extras
}

Message key: camera_id (bytes) — routes images from the same camera to
             the same partition, preserving per-camera ordering.

Usage
-----
    # Publish a single image from the CLI
    python src/serving/producer.py \\
        --image-path data/raw/Crazing/img_001.bmp \\
        --camera-id cam-line-01

    # Stress-test: publish every image in a directory tree
    python src/serving/producer.py \\
        --dir data/raw/ \\
        --camera-id cam-line-01 \\
        --delay 0.1

    # As a library
    from src.serving.producer import DefectImageProducer
    producer = DefectImageProducer()
    job_id = producer.publish("s3://camera-images/img_001.bmp")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# kafka-python (pure Python, no C extension required)
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Defaults (overridable via environment or constructor args)
# ---------------------------------------------------------------------------
DEFAULT_BOOTSTRAP_SERVERS: str = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
)
DEFAULT_TOPIC: str = os.getenv("KAFKA_TOPIC_IMAGES", "new-images")

# How long to wait for Redpanda to become available at startup
CONNECT_RETRY_SECONDS: int = 30
CONNECT_RETRY_INTERVAL: float = 2.0


# ===========================================================================
# Producer class
# ===========================================================================

class DefectImageProducer:
    """
    Thin wrapper around KafkaProducer that publishes image-reference events.

    Parameters
    ----------
    bootstrap_servers : str | list[str]
        Kafka broker address(es).  Defaults to ``KAFKA_BOOTSTRAP_SERVERS``
        env var or ``localhost:9092``.
    topic : str
        Destination topic.  Defaults to ``KAFKA_TOPIC_IMAGES`` env var or
        ``new-images``.
    acks : str
        Producer acknowledgement mode.  ``"all"`` (default) waits for all
        in-sync replicas — safest for production.  Use ``1`` for lower
        latency in dev.
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] = DEFAULT_BOOTSTRAP_SERVERS,
        topic: str = DEFAULT_TOPIC,
        acks: str = "all",
    ) -> None:
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [s.strip() for s in bootstrap_servers.split(",")]

        self.topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._producer: Optional[KafkaProducer] = None
        self._acks = acks

    # ── Lazy connection ───────────────────────────────────────────────────

    def _ensure_connected(self) -> KafkaProducer:
        """Return the KafkaProducer, creating it on first use."""
        if self._producer is not None:
            return self._producer

        deadline = time.monotonic() + CONNECT_RETRY_SECONDS
        last_exc: Exception = RuntimeError("Never attempted connection")

        while time.monotonic() < deadline:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    client_id="defect-image-producer",
                    acks=self._acks,
                    retries=5,
                    max_in_flight_requests_per_connection=1,  # preserve ordering
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    compression_type="gzip",
                    # Linger up to 5 ms to batch small messages
                    linger_ms=5,
                    # Ensure the producer buffers don't grow unbounded
                    buffer_memory=32 * 1024 * 1024,
                )
                log.info(
                    "Connected to Redpanda at %s (topic=%s)",
                    self._bootstrap_servers, self.topic,
                )
                return self._producer
            except NoBrokersAvailable as exc:
                last_exc = exc
                log.warning(
                    "Redpanda not yet available — retrying in %.1fs …",
                    CONNECT_RETRY_INTERVAL,
                )
                time.sleep(CONNECT_RETRY_INTERVAL)

        raise RuntimeError(
            f"Could not connect to Kafka brokers {self._bootstrap_servers} "
            f"after {CONNECT_RETRY_SECONDS}s: {last_exc}"
        )

    # ── Core publish ──────────────────────────────────────────────────────

    def publish(
        self,
        image_path: str,
        camera_id: str = "cam-default",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Publish a single image-reference event to Kafka.

        Parameters
        ----------
        image_path : str
            Path to the image — local filesystem or ``s3://bucket/key``.
        camera_id : str
            Logical camera / line identifier.  Used as the Kafka message key
            so all images from the same camera land on the same partition.
        metadata : dict | None
            Optional extra context (shift, line, product-id, …).
        timestamp : datetime | None
            Capture time.  Defaults to ``datetime.utcnow()``.

        Returns
        -------
        str  The ``job_id`` UUID assigned to this event.
        """
        job_id    = str(uuid.uuid4())
        ts        = (timestamp or datetime.now(timezone.utc)).isoformat()

        message: dict[str, Any] = {
            "job_id":     job_id,
            "camera_id":  camera_id,
            "image_path": str(image_path),
            "timestamp":  ts,
            "metadata":   metadata or {},
        }

        producer = self._ensure_connected()

        try:
            future = producer.send(
                self.topic,
                key=camera_id,
                value=message,
            )
            record = future.get(timeout=10)   # block until broker ACK
            log.info(
                "Published job_id=%s camera=%s path=%s → %s[%d]@%d",
                job_id, camera_id, image_path,
                record.topic, record.partition, record.offset,
            )
            return job_id

        except KafkaError as exc:
            log.error("Failed to publish image ref %s: %s", image_path, exc)
            raise

    def publish_directory(
        self,
        directory: str | Path,
        camera_id: str = "cam-default",
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        delay_s: float = 0.0,
    ) -> list[str]:
        """
        Publish all images found recursively under *directory*.

        Parameters
        ----------
        directory  : Path to scan recursively.
        camera_id  : Camera identifier applied to every event.
        extensions : Image file extensions to include.
        delay_s    : Optional sleep between publishes (rate-limiting).

        Returns
        -------
        list[str]  All job_ids published.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        image_files = sorted(
            p for p in directory.rglob("*")
            if p.suffix.lower() in extensions
        )
        if not image_files:
            log.warning("No images found under: %s", directory)
            return []

        log.info("Publishing %d images from: %s", len(image_files), directory)
        job_ids: list[str] = []

        for path in image_files:
            jid = self.publish(str(path.resolve()), camera_id=camera_id)
            job_ids.append(jid)
            if delay_s > 0:
                time.sleep(delay_s)

        return job_ids

    # ── Resource management ───────────────────────────────────────────────

    def close(self, flush_timeout: float = 5.0) -> None:
        """Flush pending messages and close the producer."""
        if self._producer is not None:
            self._producer.flush(timeout=flush_timeout)
            self._producer.close()
            self._producer = None
            log.info("Producer closed.")

    def __enter__(self) -> "DefectImageProducer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish image-reference events to the Redpanda new-images topic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image-path", "-i",
        help="Path to a single image (local or s3://bucket/key).",
    )
    group.add_argument(
        "--dir", "-d",
        help="Directory to scan recursively and publish all images.",
    )
    p.add_argument(
        "--camera-id", default="cam-default",
        help="Logical camera / sensor identifier.",
    )
    p.add_argument(
        "--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS,
        help="Comma-separated Redpanda/Kafka broker addresses.",
    )
    p.add_argument(
        "--topic", default=DEFAULT_TOPIC,
        help="Destination Kafka topic.",
    )
    p.add_argument(
        "--delay", type=float, default=0.0,
        help="Sleep (seconds) between publishes when using --dir.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    with DefectImageProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    ) as producer:

        if args.image_path:
            job_id = producer.publish(args.image_path, camera_id=args.camera_id)
            print(f"Published job_id: {job_id}")

        elif args.dir:
            job_ids = producer.publish_directory(
                args.dir,
                camera_id=args.camera_id,
                delay_s=args.delay,
            )
            print(f"Published {len(job_ids)} images.")
