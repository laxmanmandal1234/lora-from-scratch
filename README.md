# lora-from-scratch

![tests](https://github.com/laxmanmandal1234/lora-from-scratch/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Low-Rank Adaptation (LoRA)** implemented from scratch in NumPy — forward pass, hand-derived backward pass, adapter merging — plus a controlled experiment comparing LoRA ranks against head-only and full fine-tuning.

LoRA is the most widely used parameter-efficient fine-tuning (PEFT) method for large language models. This repo strips it down to its core idea so the maths and the trade-offs are easy to inspect.

## The idea

A pretrained weight matrix `W0 ∈ R^{d_in × d_out}` is **frozen**. Fine-tuning learns only a low-rank update:

```
h = x·W0 + b + (α / r) · x·A·B          A ∈ R^{d_in × r},  B ∈ R^{r × d_out},  r ≪ min(d_in, d_out)
```

- **Trainable parameters** drop from `d_in·d_out` to `r·(d_in + d_out)`.
- `B` is initialised to **zero**, so the adapted model starts *exactly* equal to the pretrained one.
- After training, `W = W0 + (α/r)·A·B` can be **merged**, giving zero extra inference cost.

### Backward pass (implemented in `lora/layers.py`)

With upstream gradient `G = ∂L/∂h` and `s = α/r`:

```
∂L/∂B = s · (xA)ᵀ G
∂L/∂A = s · xᵀ (G Bᵀ)
∂L/∂x = G W0ᵀ + s · (G Bᵀ) Aᵀ
```

Every gradient is verified against finite differences in `tests/`.

## Experiment

```bash
pip install -r requirements.txt
python experiments/run_adaptation.py      # ~3–4 minutes on a laptop CPU
```

**Setup** (synthetic, so it is fully reproducible):

- Inputs `x ~ N(0, I₃₂)`; labels (8 classes) are produced by a random "teacher" network.
- **Pretraining (task A):** 20,000 examples from teacher `T_A`.
- **Adaptation (task B):** only **1,000** examples from `T_B` — the same teacher with its first layer perturbed, i.e. a related but shifted task.
- Student model: MLP `32 → 256 → 256 → 8` (76,296 parameters). Each method trains for 60 epochs with Adam; 3 random seeds.

**Results** (task-B test accuracy on 5,000 held-out examples, mean ± std over 3 seeds):

| method | trainable params | % of model | task-B test acc |
|---|---:|---:|---:|
| zero-shot (no adaptation) | 0 | 0.00 % | 53.6 ± 1.9 |
| head-only fine-tune | 2,056 | 2.69 % | 63.3 ± 0.7 |
| LoRA r = 1 | 1,072 | 1.41 % | 58.4 ± 0.8 |
| LoRA r = 2 | 2,136 | 2.80 % | 61.5 ± 0.4 |
| LoRA r = 4 | 4,264 | 5.59 % | 66.2 ± 0.5 |
| LoRA r = 8 | 8,520 | 11.17 % | 67.4 ± 0.7 |
| LoRA r = 16 | 17,032 | 22.32 % | 68.1 ± 0.8 |
| full fine-tune | 76,296 | 100.00 % | 69.2 ± 0.8 |

**Observations**

- LoRA with **r = 8 recovers ~88 % of the gain of full fine-tuning** (+13.8 vs +15.6 points over zero-shot) while training **11 %** of the parameters.
- Accuracy grows with rank but with **diminishing returns** — most of the useful update lives in a low-dimensional subspace, which is the core hypothesis behind LoRA.
- At a matched budget (~2k parameters) head-only tuning beats LoRA r = 2 here, because the task shift is in the *first* layer and a very low-rank adapter spread across three layers is too constrained. Which layers to adapt is a real design decision.

*Caveat: this is a small synthetic study meant to illustrate the method, not a claim about LLM-scale behaviour.*

## Project structure

```
lora/layers.py               # LoRALinear: forward, backward, merge
lora/model.py                # MLP with fine-tuning modes, softmax loss, Adam, training loop
experiments/run_adaptation.py
tests/test_lora.py           # gradient checks, frozen-weight check, merge equivalence
```

## Tests

```bash
pip install pytest
pytest -q
```

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.*

## License

MIT © Laxman Mandal
