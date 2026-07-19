"""Unit tests for shared deterministic seed configuration."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from models.common.reproducibility import MAX_SEED, set_global_seed


def _draw_random_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def test_set_global_seed_repeats_python_numpy_and_torch_sequences() -> None:
    set_global_seed(42)
    first = _draw_random_values()

    set_global_seed(42)
    second = _draw_random_values()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize("deterministic", [True, False])
def test_set_global_seed_controls_deterministic_algorithm_mode(deterministic: bool) -> None:
    set_global_seed(7, deterministic=deterministic)

    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.backends.cudnn.deterministic is deterministic
    assert torch.backends.cudnn.benchmark is (not deterministic)


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_set_global_seed_rejects_out_of_range_values(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        set_global_seed(seed)


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_set_global_seed_rejects_non_integer_values(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_global_seed(seed)  # type: ignore[arg-type]
