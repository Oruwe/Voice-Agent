"""
Tests for POST /v1/documents/upload.

Auth and Moss are patched out: what's tested is the endpoint's own contract --
it never reports a document as indexed when it wasn't, and every rejection
leaves a reason in the logs (a production 422 left none).
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import app.security.token_service as token_service

AUTH = {"Authorization": "Bearer test-token"}
TEXT = b"Pump P-7 needs its seal replaced every 2000 running hours. " * 30


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MOSS_PROJECT_ID", "proj")
    monkeypatch.setenv("MOSS_PROJECT_KEY", "key")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    identity = SimpleNamespace(tenant_slug="default")
    with patch.object(token_service, "_resolve_bearer", AsyncMock(return_value=identity)):
        yield TestClient(token_service.app)


def _moss(upsert: AsyncMock):
    provider = MagicMock()
    provider.upsert_context = upsert
    return patch.object(token_service, "MossContextProvider", return_value=provider)


def test_indexed_document_reports_its_chunks(client):
    upsert = AsyncMock()
    with _moss(upsert):
        resp = client.post("/v1/documents/upload", headers=AUTH, files={"file": ("sop.txt", TEXT, "text/plain")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["chunks"] == len(upsert.await_args.kwargs["docs"]) > 1
    assert upsert.await_args.kwargs["tenant_id"] == "default"


def test_moss_failure_is_an_error_not_a_false_success(client):
    """The agent reads the knowledge base from Moss -- if that write failed,
    the document is not indexed, whatever else happened."""
    with _moss(AsyncMock(side_effect=TypeError("argument 'metadata': 'int' object is not an instance of 'str'"))):
        resp = client.post("/v1/documents/upload", headers=AUTH, files={"file": ("sop.txt", TEXT, "text/plain")})

    assert resp.status_code == 502
    assert "knowledge base" in resp.json()["detail"]


def test_document_without_text_is_rejected_with_a_logged_reason(client, caplog):
    with _moss(AsyncMock()) as moss, caplog.at_level(logging.WARNING):
        resp = client.post("/v1/documents/upload", headers=AUTH, files={"file": ("scan.txt", b"   \n  ", "text/plain")})

    assert resp.status_code == 422
    assert "no text layer" in resp.json()["detail"]
    assert "scan.txt" in caplog.text
    moss.assert_not_called()


def test_unparseable_pdf_is_rejected_with_a_logged_reason(client, caplog):
    with _moss(AsyncMock()), caplog.at_level(logging.WARNING):
        resp = client.post(
            "/v1/documents/upload", headers=AUTH,
            files={"file": ("broken.pdf", b"not really a pdf", "application/pdf")},
        )

    assert resp.status_code == 422
    assert "broken.pdf" in caplog.text and "application/pdf" in caplog.text
