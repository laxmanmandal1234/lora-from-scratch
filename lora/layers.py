"""Linear layers with optional Low-Rank Adaptation (LoRA), forward + backward in NumPy.

LoRA (Hu et al., 2021) freezes a pretrained weight matrix W0 and learns a
low-rank update:

    h = x W0 + b + (alpha / r) * x A B,      A: (d_in, r),  B: (r, d_out)

Only A and B (r * (d_in + d_out) numbers) are trained instead of
d_in * d_out.  B is initialised to zero, so at the start of fine-tuning the
adapted model is *exactly* the pretrained model.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class Param:
    """A trainable array together with its gradient."""

    def __init__(self, value: np.ndarray, name: str):
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name
        self.trainable = True

    @property
    def size(self) -> int:
        return self.value.size


class LoRALinear:
    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, name: str = "linear"):
        bound = np.sqrt(6.0 / d_in)
        self.W = Param(rng.uniform(-bound, bound, (d_in, d_out)), f"{name}.W")
        self.b = Param(np.zeros(d_out), f"{name}.b")
        self.A: Optional[Param] = None
        self.B: Optional[Param] = None
        self.scale = 0.0
        self.name = name
        self._x: Optional[np.ndarray] = None

    # --------------------------------------------------------------- LoRA API
    def add_lora(self, rank: int, alpha: float, rng: np.random.Generator) -> None:
        d_in, d_out = self.W.value.shape
        self.A = Param(rng.normal(0, 1 / np.sqrt(d_in), (d_in, rank)), f"{self.name}.lora_A")
        self.B = Param(np.zeros((rank, d_out)), f"{self.name}.lora_B")
        self.scale = alpha / rank

    def merge(self) -> None:
        """Fold the adapter into W so inference has zero extra cost."""
        if self.A is None:
            return
        self.W.value = self.W.value + self.scale * self.A.value @ self.B.value
        self.A = self.B = None
        self.scale = 0.0

    def params(self) -> Dict[str, Param]:
        out = {"W": self.W, "b": self.b}
        if self.A is not None:
            out.update(A=self.A, B=self.B)
        return out

    # ---------------------------------------------------------- forward/back
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        h = x @ self.W.value + self.b.value
        if self.A is not None:
            self._xa = x @ self.A.value
            h = h + self.scale * self._xa @ self.B.value
        return h

    def backward(self, dh: np.ndarray) -> np.ndarray:
        x = self._x
        if self.W.trainable:
            self.W.grad += x.T @ dh
        if self.b.trainable:
            self.b.grad += dh.sum(axis=0)
        dx = dh @ self.W.value.T
        if self.A is not None:
            dB_in = dh @ self.B.value.T            # (n, r)
            if self.B.trainable:
                self.B.grad += self.scale * self._xa.T @ dh
            if self.A.trainable:
                self.A.grad += self.scale * x.T @ dB_in
            dx = dx + self.scale * dB_in @ self.A.value.T
        return dx
