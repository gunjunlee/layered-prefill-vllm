# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layered_prefill import layer_group_context
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class LayeredModelCudaGraphManager(ModelCudaGraphManager):
    """Capture local prefill groups alongside the full-depth decode graphs."""

    def _init_candidates(self) -> None:
        super()._init_candidates()
        mode = self.cudagraph_mode.mixed_mode()
        if mode in self._capture_descs:
            groups = self.vllm_config.scheduler_config.num_layer_groups
            self._capture_descs[mode] = [
                replace(desc, layer_group_idx=group)
                for desc in self._capture_descs[mode]
                for group in [None, *range(groups)]
            ]


def stage_layered_intermediates(
    buffers: IntermediateTensors,
    batch: InputBatch,
    incoming: dict[str, IntermediateTensors],
) -> IntermediateTensors:
    """Gather continuations into capture buffers in the V2 batch's order."""
    staged = buffers[: batch.num_tokens_after_padding]
    for key, buffer in staged.items():
        offset = 0
        for req_id in batch.req_ids:
            tensor = incoming[req_id][key]
            end = offset + tensor.shape[0]
            buffer[offset:end].copy_(tensor, non_blocking=True)
            offset = end
        assert offset == batch.num_tokens
        buffer[offset:].zero_()
    return staged


class LayeredPrefillRunner:
    """Retain each request's activations across local groups and PP transfers."""

    def __init__(self, model_runner: "GPUModelRunner", num_groups: int) -> None:
        self.model_runner = model_runner
        self.num_groups = num_groups
        self.activations: dict[str, IntermediateTensors] = {}

    @staticmethod
    def _split_activations(
        tensors: IntermediateTensors,
        req_ids: list[str],
        num_tokens: dict[str, int],
    ) -> dict[str, IntermediateTensors]:
        activations = {}
        offset = 0
        for req_id in req_ids:
            end = offset + num_tokens[req_id]
            activations[req_id] = tensors[offset:end]
            offset = end
        return activations

    def execute_model(self, output: SchedulerOutput) -> ModelRunnerOutput | None:
        pp = get_pp_group()
        stages = output.layered_prefill_outputs
        assert stages is not None
        local = stages[pp.rank_in_group]
        for removed_id in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.activations.pop(removed_id, None)

        inputs = None
        if local.num_scheduled_tokens and (
            local.layer_group_idx != 0 or not pp.is_first_rank
        ):
            inputs = {
                req_id: self.activations.pop(req_id)
                for req_id in local.num_scheduled_tokens
            }
        with layer_group_context(local.layer_group_idx, self.num_groups):
            result = self.model_runner.execute_model(local, inputs)

        send_handles = []
        if isinstance(result, IntermediateTensors):
            state = self.model_runner.execute_model_state
            assert state is not None
            req_ids = state.input_batch.req_ids
            if local.layer_group_idx == self.num_groups - 1:
                assert not pp.is_last_rank
                # Backends may reorder requests differently on each PP rank.
                # Transfer in scheduler order, then stage in the receiver's order.
                parts = self._split_activations(
                    result, req_ids, local.num_scheduled_tokens
                )
                outgoing_tensors = {
                    key: torch.cat(
                        [parts[req_id][key] for req_id in local.num_scheduled_tokens]
                    )
                    for key in result.tensors
                }
                send_handles = pp.isend_tensor_dict(
                    outgoing_tensors, all_gather_group=get_tp_group()
                )
            else:
                # Detach from reusable graph buffers and drop graph padding.
                saved = IntermediateTensors(
                    {
                        k: v[: local.total_num_scheduled_tokens].clone()
                        for k, v in result.items()
                    }
                )
                self.activations.update(
                    self._split_activations(saved, req_ids, local.num_scheduled_tokens)
                )
            result = EMPTY_MODEL_RUNNER_OUTPUT

        # Receive at the producer's last local group, even when this rank is
        # idle. The received request runs here on the following tick.
        if not pp.is_first_rank:
            previous = stages[pp.rank_in_group - 1]
            if previous.layer_group_idx == self.num_groups - 1:
                incoming_tensors, handles, postprocess = pp.irecv_tensor_dict(
                    all_gather_group=get_tp_group()
                )
                assert incoming_tensors is not None
                for handle in handles:
                    handle.wait()
                for callback in postprocess:
                    callback()
                self.activations.update(
                    self._split_activations(
                        IntermediateTensors(incoming_tensors),
                        list(previous.num_scheduled_tokens),
                        previous.num_scheduled_tokens,
                    )
                )
        for handle in send_handles:
            handle.wait()

        completed = stages[-1]
        if (
            completed.layer_group_idx == self.num_groups - 1
            and completed.total_num_scheduled_tokens > 0
        ):
            if not pp.is_last_rank:
                self._prepare_completion(completed)
            # All PP ranks sample/receive together only after global completion.
            return None
        assert result is None or isinstance(result, ModelRunnerOutput)
        return result

    def _prepare_completion(self, completed: SchedulerOutput) -> None:
        runner = self.model_runner
        state = runner.execute_model_state
        assert state is not None
        if set(state.input_batch.req_ids) == completed.num_scheduled_tokens.keys():
            return
        runner.update_requests(completed)
        batch_state, _ = runner.gather_batch_req_state(completed, dummy_run=False)
        assert batch_state is not None
        desc = BatchExecutionDescriptor(
            CUDAGraphMode.NONE,
            completed.total_num_scheduled_tokens,
            len(completed.num_scheduled_tokens),
        )
        # Requests can be cancelled after this rank's last local group.
        # Rebuild the surviving rows so every PP rank agrees on broadcast shape.
        batch = runner.prepare_inputs(completed, batch_state, desc, 0)
        runner.execute_model_state = state._replace(input_batch=batch)
