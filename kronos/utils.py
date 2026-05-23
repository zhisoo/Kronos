"""Utility functions for Kronos time-series prediction.

Provides helpers for data preprocessing, normalization, and
post-processing of prediction outputs.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union


def normalize_series(
    data: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, dict]:
    """Normalize a 1-D or 2-D time-series array.

    Args:
        data: Input array of shape (T,) or (T, C).
        method: One of ``'minmax'`` or ``'zscore'``.
        feature_range: Target range for min-max scaling.

    Returns:
        Tuple of (normalized_array, params_dict) where *params_dict*
        contains the statistics needed to invert the transform.
    """
    data = np.asarray(data, dtype=np.float64)
    params: dict = {"method": method, "shape": data.shape}

    if method == "minmax":
        lo, hi = data.min(), data.max()
        scale = hi - lo if hi != lo else 1.0
        normed = (data - lo) / scale
        a, b = feature_range
        normed = normed * (b - a) + a
        params.update({"min": lo, "max": hi, "feature_range": feature_range})
    elif method == "zscore":
        mu, sigma = data.mean(), data.std()
        sigma = sigma if sigma != 0 else 1.0
        normed = (data - mu) / sigma
        params.update({"mean": mu, "std": sigma})
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return normed.astype(np.float32), params


def denormalize_series(data: np.ndarray, params: dict) -> np.ndarray:
    """Invert a normalization applied by :func:`normalize_series`.

    Args:
        data: Normalized array.
        params: The params dict returned by :func:`normalize_series`.

    Returns:
        Array restored to the original scale.
    """
    data = np.asarray(data, dtype=np.float64)
    method = params["method"]

    if method == "minmax":
        a, b = params["feature_range"]
        lo, hi = params["min"], params["max"]
        data = (data - a) / (b - a)
        data = data * (hi - lo) + lo
    elif method == "zscore":
        data = data * params["std"] + params["mean"]
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return data.astype(np.float32)


def prepare_context_window(
    series: Union[np.ndarray, pd.Series],
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    """Extract the most recent context window from a series.

    Args:
        series: Full historical series (1-D).
        context_length: Number of past time-steps the model expects.
        prediction_length: Number of future steps to forecast (used for
            validation only — does not affect the returned array).

    Returns:
        Array of shape ``(context_length,)`` ready to pass to the model.

    Raises:
        ValueError: If the series is shorter than *context_length*.
    """
    if isinstance(series, pd.Series):
        series = series.values

    series = np.asarray(series, dtype=np.float32)

    if len(series) < context_length:
        raise ValueError(
            f"Series length ({len(series)}) is shorter than "
            f"context_length ({context_length})."
        )

    # Take the most recent context_length steps
    return series[-context_length:]
