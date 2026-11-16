"""MLP classifier with LoRA-capable layers, a softmax loss and Adam."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .layers import LoRALinear, Param


def softmax_xent(logits: np.ndarray, y: np.ndarray):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    d = np.exp(logp)
    d[np.arange(n), y] -= 1
    return loss, d / n


class MLP:
    def __init__(self, sizes: Sequence[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [LoRALinear(a, b, rng, name=f"fc{i}")
                       for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._masks = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                mask = x > 0
                self._masks.append(mask)
                x = x * mask
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        d = dlogits
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                d = d * self._masks[i]
            d = self.layers[i].backward(d)

    def all_params(self) -> List[Param]:
        return [p for layer in self.layers for p in layer.params().values()]

    def trainable_params(self) -> List[Param]:
        return [p for p in self.all_params() if p.trainable]

    def n_trainable(self) -> int:
        return sum(p.size for p in self.trainable_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    # ----------------------------------------------------- fine-tuning modes
    def freeze_all(self) -> None:
        for p in self.all_params():
            p.trainable = False

    def setup_full_finetune(self) -> None:
        for p in self.all_params():
            p.trainable = True

    def setup_head_only(self) -> None:
        self.freeze_all()
        for p in self.layers[-1].params().values():
            p.trainable = True

    def setup_lora(self, rank: int, alpha: float, seed: int = 0, train_head_bias: bool = True) -> None:
        rng = np.random.default_rng(seed)
        self.freeze_all()
        for layer in self.layers:
            layer.add_lora(rank, alpha, rng)   # adapters are trainable by default
        if train_head_bias:
            self.layers[-1].b.trainable = True

    def merge_lora(self) -> None:
        for layer in self.layers:
            layer.merge()


class Adam:
    def __init__(self, params: List[Param], lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad[...] = 0.0

    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m[...] = self.b1 * m + (1 - self.b1) * p.grad
            v[...] = self.b2 * v + (1 - self.b2) * p.grad ** 2
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.value -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model: MLP, X, y, epochs: int, lr: float, batch: int = 64, seed: int = 0) -> float:
    opt = Adam(model.trainable_params(), lr=lr)
    rng = np.random.default_rng(seed)
    loss = float("nan")
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = idx[i:i + batch]
            loss, d = softmax_xent(model.forward(X[b]), y[b])
            opt.zero_grad()
            model.backward(d)
            opt.step()
    return float(loss)


def accuracy(model: MLP, X, y) -> float:
    return float((model.predict(X) == y).mean())
