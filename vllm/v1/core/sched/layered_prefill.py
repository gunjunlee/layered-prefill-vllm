# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, GrammarOutput, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


@dataclass
class _PrefillBatch:
    requests: list[Request]
    first: SchedulerOutput
    continuation: SchedulerOutput
    group: int = 0


class LayeredPrefillScheduler(Scheduler):
    """Drain an admitted batch through groups inside each PP rank.

    KV allocation and admission use the normal scheduler. While a batch is
    traversing the pipeline, its requests cannot be preempted or rescheduled.
    Decode-only batches use the normal full-depth execution path.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_groups = self.scheduler_config.num_layer_groups
        self._pending: deque[_PrefillBatch] = deque()
        self._stages: list[_PrefillBatch | None] = [
            None
        ] * self.parallel_config.pipeline_parallel_size
        self._defer_prefill = False

    def _update_after_schedule(self, output: SchedulerOutput) -> None:
        self._defer_prefill = any(
            self.requests[req_id].num_computed_tokens
            < self.requests[req_id].num_prompt_tokens
            for req_id in output.num_scheduled_tokens
        )
        if not self._defer_prefill:
            super()._update_after_schedule(output)
        else:
            self.finished_req_ids = set()
            self.reset_preempted_req_ids = set()

    @staticmethod
    def _select_requests(output: SchedulerOutput, req_ids: set[str]) -> SchedulerOutput:
        cached = output.scheduled_cached_reqs
        indices = [i for i, req_id in enumerate(cached.req_ids) if req_id in req_ids]
        tokens = {
            req_id: count
            for req_id, count in output.num_scheduled_tokens.items()
            if req_id in req_ids
        }
        return replace(
            output,
            scheduled_new_reqs=[
                req for req in output.scheduled_new_reqs if req.req_id in req_ids
            ],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[cached.req_ids[i] for i in indices],
                resumed_req_ids=cached.resumed_req_ids & req_ids,
                new_token_ids=[cached.new_token_ids[i] for i in indices]
                if cached.new_token_ids
                else [],
                all_token_ids={
                    req_id: ids
                    for req_id, ids in cached.all_token_ids.items()
                    if req_id in req_ids
                },
                new_block_ids=[cached.new_block_ids[i] for i in indices],
                num_computed_tokens=[cached.num_computed_tokens[i] for i in indices],
                num_output_tokens=[cached.num_output_tokens[i] for i in indices],
            ),
            num_scheduled_tokens=tokens,
            total_num_scheduled_tokens=sum(tokens.values()),
            block_table_updates=(
                {
                    req_id: blocks
                    for req_id, blocks in output.block_table_updates.items()
                    if req_id in req_ids
                }
                if output.block_table_updates is not None
                else None
            ),
        )

    def _enqueue_batch(self, output: SchedulerOutput) -> None:
        req_ids = list(output.num_scheduled_tokens)
        requests = [self.requests[req_id] for req_id in req_ids]
        continuation = CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in req_ids],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[req.num_computed_tokens for req in requests],
            num_output_tokens=[req.num_output_tokens for req in requests],
        )
        first = replace(
            output,
            num_common_prefix_blocks=[0] * len(output.num_common_prefix_blocks),
            finished_req_ids=set(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
        )
        self._pending.append(
            _PrefillBatch(
                requests,
                first,
                replace(
                    first,
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=continuation,
                ),
            )
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        housekeeping = SchedulerOutput.make_empty()
        if not self._pending and not any(self._stages):
            housekeeping = super().schedule(throttle_prefills)
            if not self._defer_prefill:
                return housekeeping
            self._enqueue_batch(housekeeping)

        finished = housekeeping.finished_req_ids | self.finished_req_ids
        preempted = (
            housekeeping.preempted_req_ids or set()
        ) | self.reset_preempted_req_ids
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

        def keep_active(job: _PrefillBatch) -> bool:
            requests = [
                req
                for req in job.requests
                if self.requests.get(req.request_id) is req
                and req.status == RequestStatus.RUNNING
            ]
            if len(requests) != len(job.requests):
                job.requests = requests
                req_ids = {req.request_id for req in requests}
                job.first = self._select_requests(job.first, req_ids)
                job.continuation = self._select_requests(job.continuation, req_ids)
            return bool(requests)

        self._pending = deque(job for job in self._pending if keep_active(job))
        for rank, job in enumerate(self._stages):
            if job is not None and not keep_active(job):
                self._stages[rank] = None
        paused = self._pause_state == PauseState.PAUSED_ALL
        if not paused and self._stages[0] is None and self._pending:
            self._stages[0] = self._pending.popleft()

        stages = []
        for job in self._stages:
            output = SchedulerOutput.make_empty()
            if job is not None and not paused:
                output = replace(
                    job.first if job.group == 0 else job.continuation,
                    layer_group_idx=job.group,
                )
            stages.append(
                replace(
                    output,
                    finished_req_ids=finished,
                    preempted_req_ids=preempted,
                    new_block_ids_to_zero=housekeeping.new_block_ids_to_zero,
                    kv_cache_block_copies=housekeeping.kv_cache_block_copies,
                )
            )

        # Move only after constructing every rank's output: a transfer at the
        # end of this iteration becomes runnable in the next iteration.
        for rank in range(len(self._stages) - 1, -1, -1):
            job = self._stages[rank]
            if job is None or paused:
                continue
            job.group += 1
            if job.group == self._num_groups:
                self._stages[rank] = None
                job.group = 0
                if rank + 1 < len(self._stages):
                    assert self._stages[rank + 1] is None
                    self._stages[rank + 1] = job

        completed = stages[-1]
        if completed.layer_group_idx == self._num_groups - 1:
            super()._update_after_schedule(completed)
        scheduled = {
            req_id: num_tokens
            for stage in stages
            for req_id, num_tokens in stage.num_scheduled_tokens.items()
        }
        return replace(
            housekeeping,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            finished_req_ids=finished,
            preempted_req_ids=preempted,
            has_structured_output_requests=completed.has_structured_output_requests,
            layered_prefill_outputs=stages,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        if scheduler_output.layered_prefill_outputs is not None:
            scheduler_output = scheduler_output.layered_prefill_outputs[-1]
        return super().get_grammar_bitmask(scheduler_output)

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        stages = scheduler_output.layered_prefill_outputs
        if stages is not None:
            scheduler_output = stages[-1]
            if scheduler_output.layer_group_idx != self._num_groups - 1:
                scheduler_output = SchedulerOutput.make_empty()
        for req_id in scheduler_output.num_scheduled_tokens:
            if (request := self.requests.get(req_id)) is not None:
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
        return super().update_from_output(scheduler_output, model_runner_output)
