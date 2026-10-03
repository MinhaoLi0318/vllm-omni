# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR-stage queue/prefill/decode/preemption fields in StageRequestStats."""

import time
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType, FinishReason
from vllm.v1.metrics.stats import IterationStats, RequestStateStats

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.outputs.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
    _native_phase_metrics,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PHASE_KEYS = ("vllm_queued_ms", "vllm_prefill_ms", "vllm_decode_ms", "vllm_num_preemptions")

_STATE_KWARGS = dict(
    request_id="r",
    external_req_id="r",
    parent_req=None,
    request_index=0,
    lora_request=None,
    prompt=None,
    prompt_token_ids=[0],
    prompt_embeds=None,
    logprobs_processor=None,
    detokenizer=None,
    max_tokens_param=None,
    arrival_time=0.0,
    queue=None,
    log_stats=False,
    stream_interval=1,
)


def _generation_processor_with_state() -> tuple[MultimodalOutputProcessor, OmniRequestState]:
    processor = MultimodalOutputProcessor(
        tokenizer=None,
        log_stats=True,
        engine_core_output_type=OutputModalityNames.AUDIO,
    )
    state = OmniRequestState(**_STATE_KWARGS, output_kind=RequestOutputKind.CUMULATIVE)
    processor.request_states[state.request_id] = state
    processor.external_req_ids[state.external_req_id].append(state.request_id)
    return processor, state


def _output(*, new_token_ids, events=None, finish_reason=None, is_segment_finished=False) -> OmniEngineCoreOutput:
    return OmniEngineCoreOutput(
        request_id="r",
        new_token_ids=new_token_ids,
        finish_reason=finish_reason,
        events=events,
        multimodal_output={OutputModalityNames.AUDIO: torch.ones(1, 4)},
        is_segment_finished=is_segment_finished,
    )


def test_phase_metrics_follow_vllm_finished_request_intervals():
    stats = RequestStateStats(
        num_preemptions=1,
        queued_ts=100.0,
        scheduled_ts=100.25,
        first_token_ts=100.75,
        last_token_ts=102.0,
    )

    metrics = _native_phase_metrics(stats)

    assert metrics["vllm_queued_ms"] == pytest.approx(250.0)
    assert metrics["vllm_prefill_ms"] == pytest.approx(500.0)
    assert metrics["vllm_decode_ms"] == pytest.approx(1250.0)
    assert metrics["vllm_num_preemptions"] == 1


def test_phase_metrics_omit_intervals_without_engine_core_events():
    # Token timestamps exist but QUEUED/SCHEDULED never arrived: queue and
    # prefill would be "first_token_ts - 0" if computed blindly.
    stats = RequestStateStats(first_token_ts=50.0, last_token_ts=50.5)

    metrics = _native_phase_metrics(stats)

    assert metrics == {"vllm_decode_ms": pytest.approx(500.0)}
    assert _native_phase_metrics(RequestStateStats()) == {}


def test_phase_metrics_for_request_finished_before_scheduling():
    # QUEUED arrived but the request ended before SCHEDULED: no interval can
    # be measured, but the event stream was observed, so 0 preemptions is real.
    stats = RequestStateStats(queued_ts=7.0)

    assert _native_phase_metrics(stats) == {"vllm_num_preemptions": 0}


def test_phase_metrics_keep_measured_zero():
    # A single generated token has a real decode interval of 0 ms.
    stats = RequestStateStats(queued_ts=10.0, scheduled_ts=10.0, first_token_ts=10.5, last_token_ts=10.5)

    metrics = _native_phase_metrics(stats)

    assert metrics["vllm_queued_ms"] == 0.0
    assert metrics["vllm_decode_ms"] == 0.0
    assert metrics["vllm_num_preemptions"] == 0


def test_engine_core_events_reach_native_metric_record():
    processor, _state = _generation_processor_with_state()

    prefill = _output(
        new_token_ids=[10],
        events=[
            EngineCoreEvent(EngineCoreEventType.QUEUED, 1.0),
            EngineCoreEvent(EngineCoreEventType.SCHEDULED, 1.25),
        ],
    )
    processor.process_outputs([prefill], engine_core_timestamp=1.75, iteration_stats=IterationStats())
    decode = _output(
        new_token_ids=[11],
        events=[
            EngineCoreEvent(EngineCoreEventType.PREEMPTED, 2.0),
            # Re-scheduling after preemption must not move the prefill start.
            EngineCoreEvent(EngineCoreEventType.SCHEDULED, 2.5),
        ],
        finish_reason=FinishReason.STOP,
    )
    iteration_stats = IterationStats()
    processor.process_outputs([decode], engine_core_timestamp=3.0, iteration_stats=iteration_stats)

    record = processor.pop_native_text_metrics("r")

    assert record["vllm_queued_ms"] == pytest.approx(250.0)
    assert record["vllm_prefill_ms"] == pytest.approx(500.0)
    assert record["vllm_decode_ms"] == pytest.approx(1250.0)
    assert record["vllm_num_preemptions"] == 1
    finished = iteration_stats.finished_requests[0]
    assert record["vllm_queued_ms"] == pytest.approx(finished.queued_time * 1000.0)
    assert record["vllm_prefill_ms"] == pytest.approx(finished.prefill_time * 1000.0)
    assert record["vllm_decode_ms"] == pytest.approx(finished.decode_time * 1000.0)
    assert record["vllm_num_preemptions"] == finished.num_preemptions


def test_native_metric_record_has_no_phase_keys_when_stats_are_off():
    processor, _state = _generation_processor_with_state()

    processor.process_outputs(
        [
            _output(
                new_token_ids=[10],
                events=[EngineCoreEvent(EngineCoreEventType.QUEUED, 1.0)],
                finish_reason=FinishReason.STOP,
            )
        ],
        engine_core_timestamp=1.5,
        iteration_stats=None,
    )

    record = processor.pop_native_text_metrics("r")
    assert not set(PHASE_KEYS) & set(record)


def test_segment_finish_does_not_record_phase_split():
    # Only the terminal finish reports the split; a streaming-input segment
    # end is not a finished vLLM request.
    processor, _state = _generation_processor_with_state()

    processor.process_outputs(
        [
            _output(
                new_token_ids=[10],
                events=[
                    EngineCoreEvent(EngineCoreEventType.QUEUED, 1.0),
                    EngineCoreEvent(EngineCoreEventType.SCHEDULED, 1.25),
                ],
                finish_reason=FinishReason.STOP,
                is_segment_finished=True,
            )
        ],
        engine_core_timestamp=1.75,
        iteration_stats=IterationStats(),
    )

    record = processor.pop_native_text_metrics("r")
    assert not set(PHASE_KEYS) & set(record)


class _NativeMetricsProcessor:
    def __init__(self, record: dict) -> None:
        self.record = record

    def pop_native_text_metrics(self, request_id: str) -> dict:
        assert request_id == "r"
        return dict(self.record)


def _text_output() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r",
        outputs=[SimpleNamespace(cumulative_token_ids=[1, 2, 3], finish_reason=FinishReason.STOP)],
    )


def _build_llm_stage_metrics(record: dict):
    client = SimpleNamespace(stage_type="llm", final_output=False, final_output_type="text")
    pool = StagePool(0, [client], output_processor=_NativeMetricsProcessor(record))
    now = time.time()
    return pool.build_stage_metrics([_text_output()], submit_ts=now, request_timestamp=now, replica_id=0)


def test_stage_metrics_carry_phase_split_from_native_record():
    metrics = _build_llm_stage_metrics(
        {
            "num_generation_tokens": 3,
            "vllm_queued_ms": 2.5,
            "vllm_prefill_ms": 30.0,
            "vllm_decode_ms": 0.0,
            "vllm_num_preemptions": 0,
        }
    )

    assert metrics.vllm_queued_ms == 2.5
    assert metrics.vllm_prefill_ms == 30.0
    assert metrics.vllm_decode_ms == 0.0
    assert metrics.vllm_num_preemptions == 0


def test_stage_metrics_leave_unobserved_phases_missing():
    metrics = _build_llm_stage_metrics({"num_generation_tokens": 3, "vllm_decode_ms": 12.0})

    assert metrics.vllm_queued_ms is None
    assert metrics.vllm_prefill_ms is None
    assert metrics.vllm_decode_ms == 12.0
    assert metrics.vllm_num_preemptions is None


def test_diffusion_stage_metrics_have_no_phase_split():
    client = SimpleNamespace(stage_type="diffusion", final_output=True, final_output_type="image")
    pool = StagePool(1, [client], output_processor=None)
    output = SimpleNamespace(request_id="req-image", _custom_output={"image": "non-empty"})

    metrics = pool.build_stage_metrics([output], submit_ts=1.0, request_timestamp=1.0, replica_id=0)

    assert all(getattr(metrics, key) is None for key in PHASE_KEYS)
