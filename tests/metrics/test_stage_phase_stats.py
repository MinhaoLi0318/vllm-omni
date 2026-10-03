# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR-stage phase split columns in the StageRequestStats summary and table."""

from __future__ import annotations

import pytest

from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics import stats as stats_module
from vllm_omni.metrics.stats import StageRequestStats, StageStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PHASE_FIELDS = ("vllm_queued_ms", "vllm_prefill_ms", "vllm_decode_ms", "vllm_num_preemptions")


def _stage_stats(**overrides) -> StageRequestStats:
    values = dict(
        batch_id=1,
        batch_size=1,
        num_tokens_in=0,
        num_tokens_out=4,
        stage_gen_time_ms=20.0,
        rx_transfer_bytes=0,
        rx_decode_time_ms=0.0,
        rx_in_flight_time_ms=0.0,
        stage_stats=StageStats(),
    )
    values.update(overrides)
    return StageRequestStats(**values)


def _summary_for(ar_stats: StageRequestStats) -> tuple[OrchestratorAggregator, dict]:
    agg = OrchestratorAggregator(num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=1)
    agg.on_stage_metrics(0, "r1", ar_stats)
    # Stage 1 stands in for a diffusion stage: it never reports the split.
    agg.on_stage_metrics(1, "r1", _stage_stats(num_tokens_out=0))
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)
    return agg, agg.build_and_log_summary()


def _capture_stage_table(monkeypatch, ar_stats: StageRequestStats) -> list[str]:
    tables: list[str] = []
    monkeypatch.setattr(stats_module.logger, "isEnabledFor", lambda _level: True)
    monkeypatch.setattr(stats_module.logger, "debug", lambda msg, *args: tables.append(msg % args))
    monkeypatch.setattr(stats_module.logger, "info", lambda *_args, **_kwargs: None)
    _summary_for(ar_stats)
    title = "[StageRequestStats [request_id=r1]]"
    return next(table for table in tables if title in table).splitlines()


def _table_row(lines: list[str], field: str) -> list[str] | None:
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == field:
            return cells[1:]
    return None


def test_phase_fields_default_to_missing():
    stats = _stage_stats()

    assert all(getattr(stats, field) is None for field in PHASE_FIELDS)


def test_summary_keeps_measured_zero_apart_from_missing():
    _agg, summary = _summary_for(
        _stage_stats(vllm_queued_ms=3.0, vllm_prefill_ms=40.0, vllm_decode_ms=900.0, vllm_num_preemptions=0)
    )

    ar_row, diffusion_row = summary["stage_table"][0]["stages"]
    assert ar_row["vllm_queued_ms"] == 3.0
    assert ar_row["vllm_prefill_ms"] == 40.0
    assert ar_row["vllm_decode_ms"] == 900.0
    assert ar_row["vllm_num_preemptions"] == 0
    assert all(diffusion_row[field] is None for field in PHASE_FIELDS)


def test_stage_table_renders_phase_rows(monkeypatch):
    lines = _capture_stage_table(
        monkeypatch,
        _stage_stats(vllm_queued_ms=3.0, vllm_prefill_ms=40.0, vllm_decode_ms=900.0, vllm_num_preemptions=2),
    )

    assert _table_row(lines, "vllm_queued_ms") == ["3.000", "None"]
    assert _table_row(lines, "vllm_prefill_ms") == ["40.000", "None"]
    assert _table_row(lines, "vllm_decode_ms") == ["900.000", "None"]
    assert _table_row(lines, "vllm_num_preemptions") == ["2", "None"]


def test_stage_table_does_not_invent_rows_when_split_is_missing(monkeypatch):
    lines = _capture_stage_table(monkeypatch, _stage_stats())

    assert _table_row(lines, "stage_gen_time_ms") is not None
    assert all(_table_row(lines, field) is None for field in PHASE_FIELDS)


def test_client_stage_metrics_snapshot_is_unchanged():
    # The per-stage snapshot feeds streaming clients and the benchmark JSON;
    # the phase split stays out of it.
    agg, _summary = _summary_for(
        _stage_stats(vllm_queued_ms=3.0, vllm_prefill_ms=40.0, vllm_decode_ms=900.0, vllm_num_preemptions=0)
    )

    snapshot = agg._build_stage_metrics_snapshot("r1")

    assert not set(PHASE_FIELDS) & set(snapshot["0"])
