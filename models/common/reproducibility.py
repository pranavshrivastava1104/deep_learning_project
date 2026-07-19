"""Deterministic random-seed configuration shared by ML workflows."""

from __future__ import annotations

import random

import numpy as np
import torch

MIN_SEED = 0
MAX_SEED = 2**32 - 1


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible process-local work.

    Args:
        seed: Integer in NumPy's supported unsigned 32-bit seed range.
        deterministic: Enable deterministic PyTorch algorithms and cuDNN behavior.
        warn_only: Warn instead of raising when PyTorch has no deterministic implementation.

    Raises:
        TypeError: If ``seed`` is not an integer or is a Boolean.
        ValueError: If ``seed`` is outside the unsigned 32-bit range.

    This controls libraries in the current process. Data-loader workers and distributed
    processes must call it with their derived worker/rank seeds as well.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not MIN_SEED <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between {MIN_SEED} and {MAX_SEED}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
