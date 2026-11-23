"""Downstream 96-step foundation-model forecasters: Chronos-2, TimesFM3, and Moirai-2-Small."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_IDS = {
    "chronos2": "amazon/chronos-2",
    "timesfm3": "google/timesfm-3.0-pytorch",
    "moirai2_small": "Salesforce/moirai-2.0-R-small",
}


class FoundationForecaster:
    """Wrapper producing 96-step median point forecasts from repaired 512-step histories."""

    def __init__(
        self,
        model_key: str,
        *,
        model_id_or_path: str | None = None,
        device: str = "cuda:0",
        horizon: int = 96,
        batch_size: int = 8,
    ):
        if model_key not in DEFAULT_MODEL_IDS:
            raise ValueError(f"Unsupported forecaster {model_key!r}; choose from {tuple(DEFAULT_MODEL_IDS)}")
        self.model_key = model_key
        self.model_id = model_id_or_path or DEFAULT_MODEL_IDS[model_key]
        self.device = device
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self._pipeline = None

    def load(self) -> "FoundationForecaster":
        import torch

        torch.set_num_threads(1)
        if self.model_key == "chronos2":
            from chronos import BaseChronosPipeline

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype=torch.float32
            )
        elif self.model_key == "timesfm3":
            from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

            revision = None
            if self.model_id == DEFAULT_MODEL_IDS["timesfm3"]:
                revision = "43046b85ec22d584a13f8098c2ed39c889e129c2"
            pipeline = TimesFM3Forecaster.from_pretrained(
                self.model_id, device=self.device, revision=revision
            )
            if pipeline.config.quantiles[pipeline.config.median_quantile_index] != 0.5:
                raise ValueError("TimesFM3 must return the median forecast")
            pipeline.model.eval().requires_grad_(False)
            self._pipeline = pipeline
        else:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

            module = Moirai2Module.from_pretrained(self.model_id)
            self._pipeline = Moirai2Forecast(
                module=module,
                prediction_length=self.horizon,
                context_length=512,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).to(self.device).eval()
        return self

    def predict(self, histories: np.ndarray, seed: int = 42) -> np.ndarray:
        """Predict [B, horizon] float64 median trajectories from finite [B, 512] histories."""
        import torch

        if self._pipeline is None:
            self.load()
        arr = np.asarray(histories, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if not np.isfinite(arr).all():
            raise ValueError("Forecaster inputs must be completely imputed finite histories")

        torch.manual_seed(int(seed))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(arr), self.batch_size):
                batch = arr[start:start + self.batch_size].astype(np.float32)
                if self.model_key == "chronos2":
                    ctx = torch.as_tensor(batch[:, None, :], dtype=torch.float32)
                    _, quantiles = self._pipeline.predict_quantiles(
                        inputs=ctx, prediction_length=self.horizon, quantile_levels=[0.5]
                    )
                    med = np.asarray(quantiles, dtype=np.float64).reshape(len(batch), self.horizon)
                elif self.model_key == "timesfm3":
                    forecasts = self._pipeline.predict_batch(
                        contexts=list(batch), horizon=self.horizon
                    )
                    med = np.stack([output.forecast for output in forecasts]).astype(np.float64)
                else:
                    past = torch.as_tensor(batch[:, :, None], dtype=torch.float32, device=self.device)
                    obs = torch.ones_like(past, dtype=torch.bool)
                    pad = torch.zeros((len(batch), batch.shape[1]), dtype=torch.bool, device=self.device)
                    preds = self._pipeline(
                        past_target=past, past_observed_target=obs, past_is_pad=pad
                    )
                    arr_p = preds.detach().cpu().numpy()
                    if arr_p.ndim == 4:
                        arr_p = arr_p[..., 0]
                    med = np.median(arr_p, axis=1).astype(np.float64)
                outputs.append(med)
        return np.concatenate(outputs, axis=0)
