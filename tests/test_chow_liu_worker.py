"""End-to-end tests for Chow-Liu web worker & Vertex task integration.

Verifies:
1. `worker.process_upload_local(..., model_type="chow_liu")` runs cleanly
   without importing TensorFlow, walks the full state machine
   (queued -> processing -> training -> scoring -> complete), ranks injected
   inattentive respondents near the top, persists `modelType == "chow_liu"`,
   and populates sorted per-column contribution percentages capped at 10.
2. `worker.process_upload_local(..., model_type="chow_liu")` honors
   cooperative cancellation before fitting.
3. `worker.callback()` forwards `modelType` from Pub/Sub messages to both
   local and Vertex handlers.
4. `train.task.train_and_predict(..., model_type="chow_liu")` runs without
   TensorFlow, writes structured outlier records with per-column
   contributions and "Original -> Predicted" diffs, and `train/task.py`'s
   CLI parser accepts `--model-type`.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd


def _make_survey_csv_bytes(n_clean: int = 120, n_outliers: int = 10, seed: int = 7) -> bytes:
    """Build a synthetic correlated survey CSV with injected random outliers."""
    rng = np.random.RandomState(seed)
    q1 = rng.choice(["a", "b"], size=n_clean, p=[0.6, 0.4])
    q2 = np.where(q1 == "a", "x", "y")
    q3 = np.where(q1 == "a", "low", "high")
    q4 = np.where(q1 == "a", "yes", "no")
    clean = pd.DataFrame({
        "row_id": [f"clean_{i}" for i in range(n_clean)],
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
    })
    # Break every pairwise dependency in the injected outlier rows
    outliers = pd.DataFrame({
        "row_id": [f"outlier_{i}" for i in range(n_outliers)],
        "q1": ["a"] * n_outliers,
        "q2": ["y"] * n_outliers,
        "q3": ["high"] * n_outliers,
        "q4": ["no"] * n_outliers,
    })
    df = pd.concat([clean, outliers], ignore_index=True)
    return df.to_csv(index=False).encode("utf-8")


def _make_mock_firestore_harness(job_id: str = "chow-liu-job"):
    """Create a stateful Firestore transaction + document harness."""
    state: dict = {"status": "queued"}
    transitions: list[str] = []

    class _FakeJobRef:
        id = job_id

        def get(self, transaction=None):
            snap = Mock()
            snap.exists = True
            snap.get = Mock(side_effect=lambda key, default=None: state.get(key, default))
            return snap

        def update(self, payload):
            state.update(payload)

    job_ref = _FakeJobRef()

    def make_transaction():
        txn = MagicMock()
        txn._read_only = False
        txn._id = b"test-txn"
        txn._max_attempts = 5
        txn.in_progress = True
        txn._write_pbs = []
        txn._clean_up = Mock()

        def _update(ref, payload):
            new_status = payload.get("status")
            if new_status is not None:
                val = new_status.value if hasattr(new_status, "value") else new_status
                transitions.append(val)
                state["status"] = val
            for k, v in payload.items():
                if k != "status":
                    state[k] = v

        txn.update.side_effect = _update
        return txn

    mock_db = MagicMock()
    mock_db.collection.return_value.document.return_value = job_ref
    mock_db.transaction.side_effect = make_transaction
    return mock_db, job_ref, state, transitions


def _make_mock_storage(csv_bytes: bytes):
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = csv_bytes
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket
    return mock_storage_client


class TestChowLiuLocalWorker:
    def test_process_upload_local_chow_liu_runs_without_tensorflow(self):
        import worker

        csv_bytes = _make_survey_csv_bytes()
        mock_db, _job_ref, state, transitions = _make_mock_firestore_harness("cl-job-1")
        mock_storage = _make_mock_storage(csv_bytes)

        # Poison sys.modules["tensorflow"] = None so any TF import raises ImportError.
        with patch.dict(sys.modules, {"tensorflow": None}):
            with patch.object(worker, "db", mock_db), \
                 patch.object(worker.storage, "Client", return_value=mock_storage), \
                 patch.object(worker, "is_job_canceled", return_value=False):
                msg = MagicMock()
                msg.message_id = "msg-cl-1"
                worker.process_upload_local(
                    "cl-job-1",
                    "test-bucket",
                    "uploads/u1/survey.csv",
                    msg,
                    model_type="chow_liu",
                )

        assert transitions == ["processing", "training", "scoring", "complete"]
        assert state["status"] == "complete"
        assert state["modelType"] == "chow_liu"
        assert "modelPreset" not in state

        outliers = state["outliers"]
        assert len(outliers) == 100
        # Scores must be sorted descending and finite positive numbers
        scores = [o["reconstruction_error"] for o in outliers]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]

        # All 10 injected broken-correlation rows must land in the top 15
        top_ids = {o["data"]["row_id"] for o in outliers[:15]}
        assert all(f"outlier_{i}" in top_ids for i in range(10))

        # Per-column contributions exist, are sorted descending, and sum to ~100%
        for rec in outliers[:10]:
            contribs = rec["contributions"]
            assert 1 <= len(contribs) <= 10
            pcts = [c["percentage"] for c in contribs]
            assert pcts == sorted(pcts, reverse=True)
            assert abs(sum(pcts) - 100.0) < 1e-4

    def test_process_upload_local_chow_liu_honors_cancellation(self):
        import worker

        csv_bytes = _make_survey_csv_bytes()
        mock_db, _job_ref, state, transitions = _make_mock_firestore_harness("cl-cancel")
        mock_storage = _make_mock_storage(csv_bytes)

        with patch.object(worker, "db", mock_db), \
             patch.object(worker.storage, "Client", return_value=mock_storage), \
             patch.object(worker, "is_job_canceled", return_value=True):
            msg = MagicMock()
            msg.message_id = "msg-cl-cancel"
            worker.process_upload_local(
                "cl-cancel",
                "test-bucket",
                "uploads/u1/survey.csv",
                msg,
                model_type="chow_liu",
            )

        assert transitions == ["processing", "training"]
        assert "outliers" not in state

    def test_callback_forwards_model_type_to_local_and_vertex(self):
        import worker

        payload = {
            "jobId": "job-cb-1",
            "bucket": "test-bucket",
            "file": "uploads/u1/survey.csv",
            "modelType": "chow_liu",
        }
        msg = MagicMock()
        msg.message_id = "msg-cb-1"
        msg.data = json.dumps(payload).encode("utf-8")

        with patch.object(worker, "check_idempotency", return_value=False), \
             patch.object(worker, "mark_message_processed"), \
             patch.object(worker, "process_upload_local") as mock_local, \
             patch.object(worker, "_processing_mode", "local"):
            worker.callback(msg)
            mock_local.assert_called_once_with(
                "job-cb-1",
                "test-bucket",
                "uploads/u1/survey.csv",
                msg,
                model_preset=None,
                model_type="chow_liu",
            )

        with patch.object(worker, "check_idempotency", return_value=False), \
             patch.object(worker, "mark_message_processed"), \
             patch.object(worker, "process_upload_vertex") as mock_vertex, \
             patch.object(worker, "_processing_mode", "vertex"):
            worker.callback(msg)
            mock_vertex.assert_called_once_with(
                "job-cb-1",
                "test-bucket",
                "uploads/u1/survey.csv",
                msg,
                model_preset=None,
                model_type="chow_liu",
            )


class TestChowLiuVertexTask:
    def test_train_and_predict_accepts_model_type_and_runs_chow_liu(self):
        from train import task

        sig = inspect.signature(task.train_and_predict)
        assert "model_type" in sig.parameters

        csv_bytes = _make_survey_csv_bytes()
        mock_storage = _make_mock_storage(csv_bytes)
        captured_sets: list[dict] = []

        mock_doc = MagicMock()
        mock_doc.set.side_effect = lambda data, merge=False: captured_sets.append(data)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc

        with patch.dict(sys.modules, {"tensorflow": None}):
            with patch.object(task, "db", mock_db), \
                 patch.object(task.storage, "Client", return_value=mock_storage):
                task.train_and_predict(
                    "vertex-cl-1",
                    "test-bucket",
                    "uploads/u1/survey.csv",
                    model_type="chow_liu",
                )

        assert len(captured_sets) == 1
        result = captured_sets[0]
        assert result["status"] == "complete"
        assert result["modelType"] == "chow_liu"
        assert len(result["outliers"]) == 100
        first = result["outliers"][0]
        assert "data" in first and "reconstruction_error" in first and "contributions" in first
        assert any(" -> " in str(v) for v in first["data"].values())

    def test_task_cli_source_accepts_model_type_flag(self):
        task_src = (Path(__file__).resolve().parents[1] / "train" / "task.py").read_text(
            encoding="utf-8"
        )
        assert "'--model-type'" in task_src or '"--model-type"' in task_src
