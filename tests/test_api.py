"""
tests/test_api.py
-----------------
Integration tests for the serving FastAPI app using FastAPI's TestClient.

These tests run *without* a real ONNX model on disk:
- A minimal ONNX graph is built programmatically in a session-scoped fixture
  so the onnxruntime.InferenceSession loads a real (but tiny) model.
- A synthetic 224×224 RGB image is created with PIL and posted to /predict.

Test coverage
-------------
1. test_health_200              – GET /health returns HTTP 200
2. test_health_schema           – /health body has expected keys + correct types
3. test_health_model_loaded     – model_loaded == True when session is active
4. test_predict_valid_response  – POST /predict returns HTTP 200
5. test_predict_response_schema – response body has all required fields
6. test_predict_confidence      – 0.0 <= confidence <= 1.0
7. test_predict_probabilities   – probabilities sum ≈ 1.0
8. test_predict_class_in_list   – predicted_class is one of CLASS_NAMES
9. test_predict_inference_ms    – inference_ms is a positive number
10. test_predict_bad_file       – non-image upload returns 422
"""

from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Make the src package importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Build a minimal ONNX model we can hand to onnxruntime
# ---------------------------------------------------------------------------

NUM_TEST_CLASSES = 6
TEST_CLASS_NAMES  = "Crazing,Inclusion,Patches,Pitted_surface,Rolled-in_scale,Scratches"


def _build_tiny_onnx(output_path: str, num_classes: int = NUM_TEST_CLASSES) -> None:
    """
    Build a trivial ONNX model: images (1,3,224,224) → logits (1, num_classes).

    Uses the onnx Python library to build the graph directly without requiring
    a running training job or a real EfficientNet checkpoint.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Weight: (num_classes, 3*224*224) — random normal
    rng    = np.random.default_rng(42)
    W_data = rng.standard_normal((num_classes, 3 * 224 * 224)).astype(np.float32)
    b_data = np.zeros(num_classes, dtype=np.float32)

    W_init = numpy_helper.from_array(W_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="b")

    # Graph: flatten → gemm
    flatten_out = helper.make_tensor_value_info("flatten_out", TensorProto.FLOAT, [None, 3 * 224 * 224])
    images_in   = helper.make_tensor_value_info("images",      TensorProto.FLOAT, [None, 3, 224, 224])
    logits_out  = helper.make_tensor_value_info("logits",      TensorProto.FLOAT, [None, num_classes])

    flatten_node = helper.make_node("Flatten", inputs=["images"],      outputs=["flatten_out"], axis=1)
    gemm_node    = helper.make_node("Gemm",    inputs=["flatten_out", "W", "b"], outputs=["logits"],
                                   transB=1)

    graph = helper.make_graph(
        [flatten_node, gemm_node],
        "tiny_defect",
        [images_in],
        [logits_out],
        initializer=[W_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def onnx_model_path() -> Generator[str, None, None]:
    """Create a temporary ONNX model file for the whole test session."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        _build_tiny_onnx(path, num_classes=NUM_TEST_CLASSES)
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def client(onnx_model_path: str) -> Generator[TestClient, None, None]:
    """
    Spin up the FastAPI app with the tiny ONNX model and the test class list.

    Environment variables are patched *before* importing the module so that
    the lifespan handler picks up the right model path.
    """
    with patch.dict(os.environ, {
        "ONNX_MODEL_PATH": onnx_model_path,
        "CLASS_NAMES":     TEST_CLASS_NAMES,
    }):
        # Re-import the module so module-level constants are re-evaluated
        if "src.serving.main" in sys.modules:
            del sys.modules["src.serving.main"]

        from src.serving.main import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A synthetic 300×300 RGB image serialised as PNG bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
        mode="RGB",
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1–3: /health tests
# ===========================================================================

class TestHealth:
    def test_health_200(self, client: TestClient) -> None:
        """GET /health must return HTTP 200 when the model is loaded."""
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_health_schema(self, client: TestClient) -> None:
        """Response body must contain all required fields with correct types."""
        body = client.get("/health").json()
        assert "status"       in body, "missing 'status'"
        assert "model_loaded" in body, "missing 'model_loaded'"
        assert "classes"      in body, "missing 'classes'"
        assert "num_classes"  in body, "missing 'num_classes'"
        assert isinstance(body["status"],       str),  "status must be str"
        assert isinstance(body["model_loaded"], bool), "model_loaded must be bool"
        assert isinstance(body["classes"],      list), "classes must be list"
        assert isinstance(body["num_classes"],  int),  "num_classes must be int"

    def test_health_model_loaded(self, client: TestClient) -> None:
        """model_loaded must be True and status must be 'ok'."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True,  "model_loaded should be True"
        assert body["status"] == "ok",        "status should be 'ok'"

    def test_health_classes_count(self, client: TestClient) -> None:
        """num_classes must match the length of the classes list."""
        body = client.get("/health").json()
        assert body["num_classes"] == len(body["classes"])
        assert body["num_classes"] == NUM_TEST_CLASSES


# ===========================================================================
# 4–9: /predict tests
# ===========================================================================

class TestPredict:
    def test_predict_200(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """POST /predict with a valid image must return HTTP 200."""
        resp = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_predict_response_schema(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Response must contain all required fields."""
        body = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()

        required = ["predicted_class", "class_index", "confidence",
                    "probabilities", "inference_ms"]
        for field in required:
            assert field in body, f"Missing field: '{field}'"

    def test_predict_confidence_in_range(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Confidence score must be in [0, 1]."""
        body = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        conf = body["confidence"]
        assert isinstance(conf, float), f"confidence must be float, got {type(conf)}"
        assert 0.0 <= conf <= 1.0, f"confidence {conf} out of [0, 1]"

    def test_predict_probabilities_sum_to_one(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Softmax probabilities must sum to ~1.0."""
        body  = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        total = sum(body["probabilities"].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_class_in_list(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """predicted_class must be one of the registered class names."""
        body = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        expected_classes = TEST_CLASS_NAMES.split(",")
        assert body["predicted_class"] in expected_classes, (
            f"'{body['predicted_class']}' is not in {expected_classes}"
        )

    def test_predict_class_index_matches_class_name(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """class_index must index into CLASS_NAMES to give predicted_class."""
        body    = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        classes = TEST_CLASS_NAMES.split(",")
        assert body["class_index"] < len(classes)
        assert classes[body["class_index"]] == body["predicted_class"]

    def test_predict_inference_ms_positive(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """inference_ms must be a positive number."""
        body = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        assert body["inference_ms"] > 0, f"inference_ms={body['inference_ms']} not positive"

    def test_predict_probabilities_keys_match_classes(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Keys in 'probabilities' must exactly match CLASS_NAMES."""
        body    = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        ).json()
        expected = set(TEST_CLASS_NAMES.split(","))
        actual   = set(body["probabilities"].keys())
        assert actual == expected, f"probability keys mismatch: {actual} != {expected}"


# ===========================================================================
# 10: Error handling
# ===========================================================================

class TestPredictErrors:
    def test_predict_non_image_returns_422(self, client: TestClient) -> None:
        """Uploading a text file as an image must return HTTP 422."""
        resp = client.post(
            "/predict",
            files={"file": ("data.txt", b"this is not an image", "text/plain")},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for non-image, got {resp.status_code}: {resp.text}"
        )

    def test_predict_empty_file_returns_422(self, client: TestClient) -> None:
        """Uploading an empty file must return HTTP 422."""
        resp = client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for empty file, got {resp.status_code}: {resp.text}"
        )

    def test_predict_jpeg_accepted(self, client: TestClient) -> None:
        """JPEG uploads must also be accepted and return HTTP 200."""
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = client.post(
            "/predict",
            files={"file": ("img.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 200, f"Expected 200 for JPEG, got {resp.status_code}"
