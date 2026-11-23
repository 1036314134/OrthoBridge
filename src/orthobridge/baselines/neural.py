"""Deep neural imputation baselines: BRITS, CSDI, SAITS, ImputeFormer, TimeMixer, HELIX.

Trained strictly on the chronological training prefix using identical 512-step windows
and deterministic 15-condition artificial masks, preserving all observed entries.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from ..protocol import MODES, RATES, WINDOW_LENGTH, mask_target, stable_seed

NEURAL_METHODS = ("brits", "csdi", "saits", "imputeformer", "timemixer", "helix")

_PYPOTS_CLASSES = {
    "saits": "SAITS",
    "brits": "BRITS",
    "csdi": "CSDI",
    "imputeformer": "ImputeFormer",
    "helix": "HELIX",
}

_MODEL_DEFAULTS = {
    "saits": {
        "n_layers": 2, "d_model": 64, "n_heads": 4, "d_k": 16, "d_v": 16,
        "d_ffn": 128, "dropout": 0.0, "attn_dropout": 0.0,
        "diagonal_attention_mask": True, "ORT_weight": 1, "MIT_weight": 1,
    },
    "brits": {"rnn_hidden_size": 64},
    "csdi": {
        "n_layers": 2, "n_heads": 4, "n_channels": 32,
        "d_time_embedding": 64, "d_feature_embedding": 16,
        "d_diffusion_embedding": 64, "n_diffusion_steps": 50,
        "target_strategy": "random", "is_unconditional": False,
        "schedule": "quad", "beta_start": 0.0001, "beta_end": 0.5,
    },
    "imputeformer": {
        "n_layers": 2, "d_input_embed": 32, "d_learnable_embed": 32, "d_proj": 16,
        "d_ffn": 128, "n_temporal_heads": 4, "dropout": 0.0, "input_dim": 1,
        "output_dim": 1, "ORT_weight": 1.0, "MIT_weight": 1.0,
    },
    "helix": {
        "n_layers": 2, "d_model": 64, "n_heads": 4, "d_pe": 16,
        "d_feature_embed": 1, "dropout": 0.1, "ORT_weight": 1.0, "MIT_weight": 1.0,
    },
    "timemixer": {
        "task_name": "imputation", "seq_len": WINDOW_LENGTH, "label_len": 0,
        "pred_len": WINDOW_LENGTH, "enc_in": 1, "dec_in": 1, "c_out": 1,
        "d_model": 64, "n_heads": 4, "e_layers": 2, "d_layers": 1, "d_ff": 128,
        "dropout": 0.0, "embed": "timeF", "freq": "h", "moving_avg": 25,
        "use_norm": True, "down_sampling_layers": 1, "down_sampling_window": 2,
        "down_sampling_method": "avg", "channel_independence": 1,
        "decomp_method": "moving_avg", "top_k": 5, "output_attention": False,
        "use_future_temporal_feature": False,
    },
}


def _load_timemixer_class():
    base_dir = Path(__file__).resolve().parent / "timemixer"
    meta = json.loads((base_dir / "source.json").read_text(encoding="utf-8"))
    src_dir = str(base_dir / "source" / meta["revision"])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    module = importlib.import_module("models.TimeMixer")
    return module.Model


def _training_mask(window_2d: np.ndarray, window_id: int, epoch: int, seed: int) -> np.ndarray:
    result = np.zeros(window_2d.shape, dtype=bool)
    cond = (int(window_id) + int(epoch)) % 15
    rate = (0.1, 0.2, 0.3, 0.4, 0.5)[cond % 5]
    mode = MODES[cond // 5]
    result[:, 0] = mask_target(
        window_2d[:, 0], mode, rate, stable_seed("train", int(window_id), int(epoch), int(seed))
    )
    return result


class NeuralBaselineImputer:
    """Unified training-prefix neural baseline runner matching the 512-step UV protocol."""

    def __init__(
        self,
        method: str,
        *,
        context_length: int = WINDOW_LENGTH,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ):
        if method not in NEURAL_METHODS:
            raise ValueError(f"Unsupported neural baseline: {method!r}")
        self.method = method
        self.length = int(context_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.physical_batch = 2 if method == "helix" else self.batch_size
        self.learning_rate = float(learning_rate)
        self.device = str(device)
        self.seed = int(seed)
        self.center = np.zeros(1, dtype=np.float64)
        self.scale = np.ones(1, dtype=np.float64)
        self.model = None
        self.fitted = False

    def _seed_rng(self, seed: int) -> None:
        import torch

        torch.manual_seed(int(seed))
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _build_model(self) -> None:
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        self._seed_rng(self.seed)
        params = deepcopy(_MODEL_DEFAULTS[self.method])
        if self.method == "timemixer":
            params["seq_len"] = self.length
            params["pred_len"] = self.length
            cls = _load_timemixer_class()
            self.model = cls(SimpleNamespace(**params)).to(self.device)
        else:
            from pypots import imputation

            cls = getattr(imputation, _PYPOTS_CLASSES[self.method])
            backend = cls(
                n_steps=self.length,
                n_features=1,
                batch_size=self.batch_size,
                epochs=self.epochs,
                patience=None,
                num_workers=0,
                device=self.device,
                saving_path=None,
                model_saving_strategy=None,
                verbose=False,
                **params,
            )
            self.model = backend.model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _make_inputs(self, raw_batch: np.ndarray, hidden: np.ndarray | None = None) -> dict:
        torch = self.torch
        finite = np.isfinite(raw_batch)
        hid = np.zeros_like(finite) if hidden is None else np.asarray(hidden, dtype=bool)
        visible = finite & ~hid
        y = torch.as_tensor(np.nan_to_num(raw_batch), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(visible, dtype=torch.float32, device=self.device)
        x = y * mask
        inputs = {
            "X": x,
            "missing_mask": mask,
            "X_ori": y,
            "indicating_mask": torch.as_tensor(hid, dtype=torch.float32, device=self.device),
        }
        if self.method == "brits":
            def _recurrent(v, m):
                delta = torch.zeros_like(m)
                for step in range(1, m.shape[1]):
                    delta[:, step] = 1.0 + (1.0 - m[:, step - 1]) * delta[:, step - 1]
                return {"X": v, "missing_mask": m, "deltas": delta}

            inputs["forward"] = _recurrent(x, mask)
            inputs["backward"] = _recurrent(x.flip(1), mask.flip(1))
        elif self.method == "csdi":
            inputs = {k: val.transpose(1, 2) for k, val in inputs.items() if k != "missing_mask"}
            inputs["cond_mask"] = mask.transpose(1, 2)
            inputs["observed_tp"] = torch.arange(raw_batch.shape[1], device=self.device).float()[None].repeat(len(raw_batch), 1)
        return inputs

    def fit(self, prefix_1d: np.ndarray, starts: np.ndarray) -> "NeuralBaselineImputer":
        prefix = np.asarray(prefix_1d, dtype=np.float64).reshape(-1, 1)
        self.center = np.nanmean(prefix, axis=0)
        self.scale = np.maximum(np.nanstd(prefix, axis=0), 1e-6)
        norm = (prefix - self.center) / self.scale
        windows = np.stack([norm[int(s):int(s) + self.length] for s in starts]).astype(np.float32)
        self._build_model()
        for epoch in range(self.epochs):
            self.model.train()
            hidden = np.stack([_training_mask(w, int(st), epoch, self.seed) for w, st in zip(windows, starts)])
            order = np.random.default_rng(stable_seed("neural_order", self.seed, epoch)).permutation(len(windows))
            for begin in range(0, len(order), self.batch_size):
                ids = order[begin:begin + self.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                for slot in range(0, len(ids), self.physical_batch):
                    part = ids[slot:slot + self.physical_batch]
                    inp = self._make_inputs(windows[part], hidden[part])
                    if self.method == "timemixer":
                        pred = self.model(inp["X"], None, None, None, mask=inp["missing_mask"])
                        ind = inp["indicating_mask"].bool()
                        loss = (pred[ind] - inp["X_ori"][ind]).square().mean()
                    else:
                        loss = self.model(inp, calc_criterion=True)["loss"]
                    (loss * (len(part) / len(ids))).backward()
                self.optimizer.step()
        self.model.eval()
        self.fitted = True
        return self

    def impute(self, query_1d: np.ndarray, request_seed: int = 42) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Fit the neural baseline before calling impute()")
        raw = np.asarray(query_1d, dtype=np.float64).reshape(-1, 1)
        obs = np.isfinite(raw)
        if obs.all():
            return raw[:, 0].copy()
        self._seed_rng(int(request_seed))
        norm = ((raw - self.center) / self.scale).astype(np.float32)[None]
        with self.torch.inference_mode():
            self.model.eval()
            inp = self._make_inputs(norm)
            if self.method == "timemixer":
                pred = self.model(inp["X"], None, None, None, mask=inp["missing_mask"]).detach().cpu().numpy()
            elif self.method == "csdi":
                samples = self.model(inp, n_sampling_times=20)["imputation"].detach().cpu().numpy()
                pred = np.median(samples, axis=1)
            else:
                pred = self.model(inp)["imputation"].detach().cpu().numpy()
        out = np.asarray(pred, dtype=np.float64)[0] * self.scale + self.center
        out[obs] = raw[obs]
        return out[:, 0].copy()
