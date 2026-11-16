import copy

import numpy as np
import pytest

from lora import MLP, LoRALinear, softmax_xent, train, accuracy

rng = np.random.default_rng(0)


def numeric_grad(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for i in np.ndindex(arr.shape):
        old = arr[i]
        arr[i] = old + eps
        fp = f()
        arr[i] = old - eps
        fm = f()
        arr[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def make_model_with_lora():
    m = MLP([4, 6, 3], seed=0)
    m.setup_lora(rank=2, alpha=4, seed=1)
    for layer in m.layers:                       # make B non-zero so grads flow to A
        layer.B.value = rng.normal(size=layer.B.value.shape)
    return m


def test_lora_starts_identical_to_base():
    base = MLP([5, 8, 3], seed=0)
    x = rng.normal(size=(7, 5))
    before = base.forward(x)
    adapted = copy.deepcopy(base)
    adapted.setup_lora(rank=4, alpha=8)
    np.testing.assert_allclose(adapted.forward(x), before)


@pytest.mark.parametrize("which", ["A", "B", "b"])
def test_gradients_match_finite_differences(which):
    m = make_model_with_lora()
    m.layers[-1].b.trainable = True
    x = rng.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0])

    loss_fn = lambda: softmax_xent(m.forward(x), y)[0]
    _, d = softmax_xent(m.forward(x), y)
    for p in m.all_params():
        p.grad[...] = 0
    m.backward(d)

    for layer in m.layers:
        p = getattr(layer, which)
        if not p.trainable:
            continue
        np.testing.assert_allclose(p.grad, numeric_grad(loss_fn, p.value), rtol=1e-5, atol=1e-8)


def test_frozen_weights_do_not_change():
    m = make_model_with_lora()
    W_before = [l.W.value.copy() for l in m.layers]
    X = rng.normal(size=(64, 4))
    y = rng.integers(0, 3, 64)
    train(m, X, y, epochs=3, lr=1e-2)
    for l, W in zip(m.layers, W_before):
        np.testing.assert_array_equal(l.W.value, W)


def test_merge_preserves_outputs():
    m = make_model_with_lora()
    x = rng.normal(size=(10, 4))
    before = m.forward(x)
    m.merge_lora()
    assert all(l.A is None for l in m.layers)
    np.testing.assert_allclose(m.forward(x), before, atol=1e-10)


def test_parameter_count():
    layer = LoRALinear(100, 50, rng)
    layer.add_lora(rank=4, alpha=8, rng=rng)
    assert layer.A.size + layer.B.size == 4 * (100 + 50)


def test_lora_can_learn():
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = MLP([6, 32, 2], seed=0)
    m.setup_lora(rank=4, alpha=8)
    train(m, X, y, epochs=40, lr=1e-2)
    assert accuracy(m, X, y) > 0.9
