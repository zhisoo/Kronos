"""Core Kronos model wrapper for time-series prediction.

This module provides a unified interface for loading and running
the Kronos forecasting model on stock market data.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union


class KronosPredictor:
    """Wrapper around the Kronos time-series forecasting model.

    Handles model loading, input preparation, and prediction generation
    for stock price forecasting tasks.

    Args:
        model_path: Path to a local model checkpoint, or a HuggingFace
                    model identifier (e.g. 'amazon/chronos-t5-small').
        device: Torch device string ('cpu', 'cuda', 'mps'). Defaults to
                auto-detection.
        prediction_length: Number of future time steps to forecast.
        num_samples: Number of sample paths to draw for probabilistic
                     forecasting.
    """

    DEFAULT_MODEL = "amazon/chronos-t5-small"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        prediction_length: int = 30,
        num_samples: int = 20,
    ) -> None:
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.device = device or self._detect_device()
        self._pipeline = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Return the best available torch device string."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> "KronosPredictor":
        """Load the Chronos pipeline from *model_path*.

        Returns self to allow method chaining::

            predictor = KronosPredictor().load()
        """
        try:
            from chronos import ChronosPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'chronos' package is required. "
                "Install it with: pip install chronos-forecasting"
            ) from exc

        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        return self

    @property
    def is_loaded(self) -> bool:
        """True if the underlying pipeline has been initialised."""
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        context: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a probabilistic forecast from *context*.

        Args:
            context: 1-D array-like of historical close prices.

        Returns:
            Tuple of (low, median, high) each shaped
            ``(prediction_length,)``, representing the 10th, 50th, and
            90th percentile forecasts.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call .load() before .predict()."
            )

        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)

        forecast = self._pipeline.predict(
            context=context.unsqueeze(0),  # batch dimension
            prediction_length=self.prediction_length,
            num_samples=self.num_samples,
        )  # shape: (1, num_samples, prediction_length)

        samples = forecast[0].numpy()  # (num_samples, prediction_length)
        low = np.percentile(samples, 10, axis=0)
        median = np.percentile(samples, 50, axis=0)
        high = np.percentile(samples, 90, axis=0)
        return low, median, high
