# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_layer_group: ContextVar[tuple[int, int] | None] = ContextVar(
    "layered_prefill_group", default=None
)


@contextmanager
def layer_group_context(group_idx: int | None, num_groups: int) -> Iterator[None]:
    group = None if group_idx is None else (group_idx, num_groups)
    token = _layer_group.set(group)
    try:
        yield
    finally:
        _layer_group.reset(token)


def get_layer_group_range(start_layer: int, end_layer: int) -> tuple[int, int]:
    """Split the local PP partition into contiguous, nonempty groups."""
    group = _layer_group.get()
    if group is None:
        return start_layer, end_layer
    idx, count = group
    num_layers = end_layer - start_layer
    if not 0 <= idx < count <= num_layers:
        raise ValueError("Layer groups must be nonempty and lie within the PP rank.")
    size, remainder = divmod(num_layers, count)
    start = start_layer + idx * size + min(idx, remainder)
    return start, start + size + (idx < remainder)
