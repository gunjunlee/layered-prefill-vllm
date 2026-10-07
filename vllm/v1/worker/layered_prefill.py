# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class LayeredPrefillRunner:
    """Retain activations locally until a request finishes this PP partition."""

    def __init__(self, model_runner: "GPUModelRunner", num_groups: int) -> None:
        self.model_runner = model_runner
        self.num_groups = num_groups
        self.activations: dict[str, IntermediateTensors] = {}

    def execute_model(self, output: SchedulerOutput) -> ModelRunnerOutput | None:
        pp = get_pp_group()
        stages = output.layered_prefill_outputs
        assert stages is not None
        local = stages[pp.rank_in_group]
        for removed_id in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.activations.pop(removed_id, None)

        req_id = next(iter(local.num_scheduled_tokens), None)
        inputs = self.activations.pop(req_id, None) if req_id is not None else None
        if req_id is not None and (local.layer_group_idx != 0 or not pp.is_first_rank):
            assert inputs is not None, f"Missing layered prefill activation: {req_id}"
        result = self.model_runner.execute_model(local, inputs)

        send_handles = []
        if isinstance(result, IntermediateTensors):
            assert req_id is not None
            # Runner input and residual buffers can be reused on the next tick.
            saved = IntermediateTensors({k: v.clone() for k, v in result.items()})
            if local.layer_group_idx == self.num_groups - 1:
                assert not pp.is_last_rank
                send_handles = pp.isend_tensor_dict(
                    saved.tensors, all_gather_group=get_tp_group()
                )
            else:
                self.activations[req_id] = saved
            result = EMPTY_MODEL_RUNNER_OUTPUT

        # Receive at the producer's last local group, even when this rank is
        # idle. The received request runs here on the following tick.
        if not pp.is_first_rank:
            previous = stages[pp.rank_in_group - 1]
            if previous.layer_group_idx == self.num_groups - 1:
                incoming_id = next(iter(previous.num_scheduled_tokens))
                tensors, handles, postprocess = pp.irecv_tensor_dict(
                    all_gather_group=get_tp_group()
                )
                assert tensors is not None
                for handle in handles:
                    handle.wait()
                for callback in postprocess:
                    callback()
                self.activations[incoming_id] = IntermediateTensors(tensors)
        for handle in send_handles:
            handle.wait()
        assert result is None or isinstance(result, ModelRunnerOutput)
        return result
