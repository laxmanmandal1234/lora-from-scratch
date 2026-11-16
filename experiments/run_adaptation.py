"""Pretrain on task A, then adapt to a related task B with different methods.

Setup (fully synthetic, so it runs in a few minutes on a laptop CPU):

* Inputs x ~ N(0, I_32). Labels come from a random "teacher" MLP.
* Task A: 20,000 examples labelled by teacher T_A  -> used for pretraining.
* Task B: only 1,000 examples labelled by T_B, where T_B is T_A with its
  first-layer weights perturbed (a related but different task, like a new
  domain).  Small data is the regime where parameter-efficient methods matter.

Methods compared on task B:
  zero-shot (pretrained model, no adaptation), head-only fine-tuning,
  LoRA with ranks 1/2/4/8/16, full fine-tuning.

    python experiments/run_adaptation.py
"""

import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lora import MLP, accuracy, train  # noqa: E402

D, H, C = 32, 256, 8


def teacher_labels(X, W1, W2):
    return (np.tanh(X @ W1) @ W2).argmax(axis=1)


def make_tasks(seed: int):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1 / np.sqrt(D), (D, 64))
    W2 = rng.normal(0, 1 / np.sqrt(64), (64, C))
    W1_b = W1 + 0.6 * rng.normal(0, 1 / np.sqrt(D), (D, 64))   # task shift
    XA = rng.normal(size=(20000, D))
    XB = rng.normal(size=(1000, D))
    XB_test = rng.normal(size=(5000, D))
    return (XA, teacher_labels(XA, W1, W2),
            XB, teacher_labels(XB, W1_b, W2),
            XB_test, teacher_labels(XB_test, W1_b, W2))


def run(seed: int):
    XA, yA, XB, yB, XT, yT = make_tasks(seed)
    base = MLP([D, H, H, C], seed=seed)
    train(base, XA, yA, epochs=15, lr=1e-3, seed=seed)

    results = {"zero-shot": (0, accuracy(base, XT, yT))}

    def adapt(name, setup, lr):
        m = copy.deepcopy(base)
        setup(m)
        n = m.n_trainable()
        train(m, XB, yB, epochs=60, lr=lr, batch=32, seed=seed)
        results[name] = (n, accuracy(m, XT, yT))

    adapt("head-only", lambda m: m.setup_head_only(), 3e-3)
    for r in [1, 2, 4, 8, 16]:
        adapt(f"LoRA r={r}", lambda m, r=r: m.setup_lora(rank=r, alpha=2 * r, seed=seed), 3e-3)
    adapt("full fine-tune", lambda m: m.setup_full_finetune(), 1e-3)
    return results


def main():
    seeds = [0, 1, 2]
    t0 = time.perf_counter()
    all_res = [run(s) for s in seeds]
    total = sum(p.size for p in MLP([D, H, H, C]).all_params())
    print(f"model: MLP {D}-{H}-{H}-{C}, {total:,} parameters; {len(seeds)} seeds; "
          f"{time.perf_counter() - t0:.0f}s\n")
    print(f"| method | trainable params | % of model | task-B test acc |")
    print(f"|---|---:|---:|---:|")
    for name in all_res[0]:
        n = all_res[0][name][0]
        accs = np.array([r[name][1] for r in all_res]) * 100
        print(f"| {name} | {n:,} | {100 * n / total:.2f}% | {accs.mean():.1f} +- {accs.std():.1f} |")


if __name__ == "__main__":
    main()
