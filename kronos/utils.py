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
    series = np.asarray(series, dtype=np.float32).ravel()

    if len(series) < context_length:
        raise ValueError(
            f"Series length {len(series)} is shorter than the required "
            f"context_length {context_length}."
        )

    if prediction_length < 1:
        raise ValueError("prediction_length must be >= 1.")

    return series[-context_length:]


def build_forecast_index(
    last_date: pd.Timestamp,
    prediction_length: int,
    freq: str = "B",
) -> pd.DatetimeIndex:
    """Build a DatetimeIndex for the forecast horizon.

    Args:
        last_date: The last observed date in the historical series.
        prediction_length: Number of future periods to generate.
        freq: Pandas offset alias, e.g. ``'B'`` (business day), ``'D'``,
              ``'H'``, ``'T'``.

    Returns:
        A :class:`pd.DatetimeIndex` of length *prediction_length* starting
        one period after *last_date*.
    """
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),  # type: ignore[operator]
        periods=prediction_length,
        freq=freq,
    )


def forecast_to_dataframe(
    forecast: np.ndarray,
    index: pd.DatetimeIndex,
    quantiles: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Convert a raw forecast array to a tidy DataFrame.

    Args:
        forecast: Array of shape ``(prediction_length,)`` for a point
            forecast, or ``(num_quantiles, prediction_length)`` for a
            quantile forecast.
        index: DatetimeIndex of length *prediction_length*.
        quantiles: Sequence of quantile levels matching the first axis of
            *forecast* when it is 2-D.  Ignored for point forecasts.

    Returns:
        DataFrame indexed by *index*.
    """
    forecast = np.asarray(forecast)

    if forecast.ndim == 1:
        return pd.DataFrame({"forecast": forecast}, index=index)

    if forecast.ndim == 2:
        if quantiles is None:
            quantiles = tuple(range(forecast.shape[0]))
        cols = {f"q{q:.2f}": forecast[i] for i, q in enumerate(quantiles)}
        return pd.DataFrame(cols, index=index)

    raise ValueError("forecast must be 1-D or 2-D.")
