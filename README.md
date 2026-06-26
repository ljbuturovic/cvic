# cvic

[![tests](https://github.com/ljbuturovic/cvic/actions/workflows/test.yml/badge.svg)](https://github.com/ljbuturovic/cvic/actions/workflows/test.yml)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Local, automated hyperparameter search for image classifiers — from dataset to tuned model with one command, distributed across your local GPUs.

cvic uses off-the-shelf models and packages, so you won't get SOTA performance. But it can get surprisingly close, with almost zero effort. Useful as a baseline, or for experimentation with architectures and GPUs.

Built on [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), [Optuna](https://optuna.org/), and [timm](https://github.com/huggingface/pytorch-image-models). Requires Python ≥ 3.12.

It ships two commands:

- **`cvic`** — k-fold cross-validation hyperparameter search
- - **`tunic`** — hold-out hyperparameter tuning (single train/validation split)

## Install

```bash
pipx install cvic
```

or with uv:

```bash
uv tool install cvic
```

## Run from source

The project is fully managed by [uv](https://docs.astral.sh/uv/) with a committed `uv.lock`, so the exact dependency versions are reproducible across machines. You need an NVIDIA GPU with a reasonably recent driver to use CUDA; the PyTorch wheels bundle their own CUDA runtime, so **no system CUDA toolkit installation is required** and you do not pick a CUDA version — `uv` resolves the right wheel for your platform automatically.

```bash
git clone https://github.com/ljbuturovic/cvic.git
cd cvic
uv sync                       # creates .venv and installs the locked dependencies
source .venv/bin/activate     # now `cvic` and `tunic` are on your PATH
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Then run the commands directly (no `uv run` prefix needed once the venv is activated):

```bash
cvic --smoke-test
tunic --smoke-test
```

To run the test suite:

```bash
pytest tests/ -k "not test_smoke"
```

## Quick start

**Hold-out tuning:**
```bash
tunic --data /path/to/dataset --model resnet50 --n_trials 30 --epochs 30 --output results.json
```

**Cross-validation tuning:**
```bash
cvic --data /path/to/dataset --model resnet50 --n-trials 30 --epochs 30 --folds 5
```

**Train final model from tuning results:**
```bash
tunic --final results.json --data /path/to/dataset --epochs 50 --amp
```

**Smoke test (synthetic data, no dataset needed):**
```bash
tunic --smoke-test
cvic --smoke-test
```

## Dataset format

The dataset format is auto-detected:

- **ImageFolder** — standard `split/class/image.ext` layout
- **WebDataset** — sharded TAR files; detected when `wds/dataset_info.json` exists

## tunic — hold-out hyperparameter search

```
tunic --data PATH --model MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Dataset root (ImageFolder or WebDataset) |
| `--model` | required | Any timm model name |
| `--n_trials` | 80 | Number of Optuna trials |
| `--epochs` | 30 | Training epochs per trial (also used for `--final`) |
| `--tune-metric` | `val_auroc` | Metric for trial selection and pruning |
| `--training_fraction` | 1.0 | Fraction of training data (val always uses 1.0) |
| `--batch-size` | 32 | Batch size per trial |
| `--amp` | | Enable automatic mixed precision |
| `--resume` | | Warm-start from a previous experiment directory |
| `--final` | | Skip tuning; train final model from results JSON |
| `--combine` | | Train final model on train+val combined |
| `--final-model` | `tunic_final.pt` | Output path for final model weights |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--smoke-test` | | Quick end-to-end test with synthetic data |

## cvic — cross-validation hyperparameter search

```
cvic --data PATH --model MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Dataset root (ImageFolder or WebDataset) |
| `--model` | required | Any timm model name |
| `--n-trials` | | Number of Optuna trials |
| `--epochs` | | Training epochs per trial |
| `--folds` | | Number of cross-validation folds |
| `--repeats` | | Repeated cross-validation runs |
| `--stratified` | | Use stratified folds |
| `--tune-metric` | `val_auroc` | Metric for trial selection |
| `--batch-size` | 32 | Batch size per trial |
| `--test-data` | | Held-out test set for final evaluation |
| `--amp` | | Enable automatic mixed precision |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--smoke-test` | | Quick end-to-end test with synthetic data |

Run `cvic --help` / `tunic --help` for the full list of flags.

## Search space

| Parameter | Range |
|---|---|
| Optimizer | AdamW, SGD |
| Learning rate | 1e-5 – 1e-1 (log) |
| Weight decay | 1e-6 – 1e-1 (log) |
| Label smoothing | 0 – 0.3 |
| Dropout rate | 0 – 0.5 |
| RandAugment magnitude | 1 – 15 |
| RandAugment num ops | 1 – 4 |
| Mixup alpha | 0 – 0.5 |
| CutMix alpha | 0 – 1.0 |

Override any part with a YAML file via `--search-space`.

## License

MIT
