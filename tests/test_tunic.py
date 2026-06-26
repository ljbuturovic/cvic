"""Unit and integration tests for tunic."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

def test_detect_format_s3():
    from cvic.tunic import _detect_format
    assert _detect_format("s3://my-bucket/dataset") == "webdataset"


def test_detect_format_imagefolder(tmp_path):
    from cvic.tunic import _detect_format
    assert _detect_format(str(tmp_path)) == "imagefolder"


def test_detect_format_webdataset(tmp_path):
    from cvic.tunic import _detect_format
    (tmp_path / "wds").mkdir()
    (tmp_path / "wds" / "dataset_info.json").write_text("{}")
    assert _detect_format(str(tmp_path)) == "webdataset"


# ---------------------------------------------------------------------------
# load_search_space_overrides
# ---------------------------------------------------------------------------

def test_load_search_space_overrides(tmp_path):
    from cvic.common_cvic import load_search_space_overrides
    f = tmp_path / "ss.yaml"
    f.write_text("lr: [0.001, 0.01]\noptimizer: [AdamW]\n")
    result = load_search_space_overrides(str(f))
    assert result["lr"] == [0.001, 0.01]
    assert result["optimizer"] == ["AdamW"]


def test_load_search_space_overrides_empty(tmp_path):
    from cvic.common_cvic import load_search_space_overrides
    f = tmp_path / "empty.yaml"
    f.write_text("")
    assert load_search_space_overrides(str(f)) == {}


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------

def test_build_scheduler_start_step_resumes_schedule():
    import torch
    from cvic.common_cvic import build_scheduler

    epochs = 4
    steps_per_epoch = 3
    start_step = 5
    base_lr = 0.1

    model_a = torch.nn.Linear(1, 1)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=base_lr)
    sched_a = build_scheduler(opt_a, epochs, steps_per_epoch, warmup_epochs=1)
    for _ in range(start_step):
        opt_a.step()
        sched_a.step()

    model_b = torch.nn.Linear(1, 1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=base_lr)
    sched_b = build_scheduler(
        opt_b, epochs, steps_per_epoch, warmup_epochs=1, start_step=start_step
    )

    assert opt_b.param_groups[0]["lr"] == pytest.approx(opt_a.param_groups[0]["lr"])

    opt_a.step()
    sched_a.step()
    opt_b.step()
    sched_b.step()
    assert opt_b.param_groups[0]["lr"] == pytest.approx(opt_a.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# _compute_auroc
# ---------------------------------------------------------------------------

def test_compute_auroc_binary_perfect():
    from cvic.common_cvic import _compute_auroc
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    assert _compute_auroc(probs, labels) == 1.0


def test_compute_auroc_binary_random():
    from cvic.common_cvic import _compute_auroc
    probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 0, 1, 1])
    auroc = _compute_auroc(probs, labels)
    assert 0.0 <= auroc <= 1.0


def test_compute_auroc_multiclass():
    from cvic.common_cvic import _compute_auroc
    # 3-class, each class perfectly predicted
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    labels = np.array([0, 1, 2, 0])
    auroc = _compute_auroc(probs, labels)
    assert auroc == pytest.approx(1.0)


def test_compute_auroc_single_class_returns_nan():
    from cvic.common_cvic import _compute_auroc
    probs = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 0, 0])
    assert np.isnan(_compute_auroc(probs, labels))


# ---------------------------------------------------------------------------
# build_transforms
# ---------------------------------------------------------------------------

def test_build_transforms_train_output_shape():
    from cvic.common_cvic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, randaug_magnitude=0, is_train=True)
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    tensor = tf(img)
    assert tensor.shape == (3, 64, 64)


def test_build_transforms_val_output_shape():
    from cvic.common_cvic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, is_train=False)
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    tensor = tf(img)
    assert tensor.shape == (3, 64, 64)


def test_build_transforms_normalized():
    from cvic.common_cvic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, is_train=False)
    img = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8))
    tensor = tf(img)
    # After ImageNet normalization values should not be in [0, 1]
    assert tensor.min().item() < 0.5


# ---------------------------------------------------------------------------
# make_stratified_split
# ---------------------------------------------------------------------------

def test_make_stratified_split_sizes(tmp_path):
    from torchvision import datasets
    from cvic.common_cvic import make_stratified_split
    # Build a tiny imagefolder with 3 classes, 10 images each
    for cls in ["a", "b", "c"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(10):
            from PIL import Image
            Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    ds = datasets.ImageFolder(str(tmp_path))
    train, val = make_stratified_split(ds, val_fraction=0.2, seed=42)
    assert len(train) + len(val) == 30
    assert len(val) == pytest.approx(6, abs=3)  # ~20% of 30


def test_make_stratified_split_reproducible(tmp_path):
    from torchvision import datasets
    from cvic.common_cvic import make_stratified_split
    for cls in ["a", "b"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(10):
            from PIL import Image
            Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    ds = datasets.ImageFolder(str(tmp_path))
    _, val1 = make_stratified_split(ds, val_fraction=0.2, seed=42)
    _, val2 = make_stratified_split(ds, val_fraction=0.2, seed=42)
    assert list(val1.indices) == list(val2.indices)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def test_cli_dash_args(monkeypatch):
    """Verify that dashed args (--n-trials, --training-fraction) are accepted."""
    import sys
    from cvic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic",
        "--data", "/tmp",
        "--model", "resnet18",
        "--n-trials", "5",
        "--training-fraction", "0.1",
        "--epochs", "2",
    ])
    args = parse_args()
    assert args.n_trials == 5
    assert args.training_fraction == pytest.approx(0.1)


def test_cli_amp_flag(monkeypatch):
    import sys
    from cvic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18", "--amp",
    ])
    args = parse_args()
    assert args.amp is True


def test_cli_prefix(monkeypatch):
    import sys
    from cvic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18", "--prefix", "myrun",
    ])
    args = parse_args()
    assert args.prefix == "myrun"


def test_cli_shuffle(monkeypatch):
    import sys
    from cvic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18", "--shuffle", "99",
    ])
    args = parse_args()
    assert args.shuffle == 99


def test_cli_shuffle_default_is_none(monkeypatch):
    import sys
    from cvic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18",
    ])
    args = parse_args()
    assert args.shuffle is None


# ---------------------------------------------------------------------------
# _build_loaders — no-val-dir behavior
# ---------------------------------------------------------------------------

def _make_train_only_dir(tmp_path, classes=("a", "b", "c"), n_per_class=20):
    from PIL import Image
    for cls in classes:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(n_per_class):
            Image.fromarray(
                np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            ).save(d / f"{i}.jpg")


def test_build_loaders_no_val_dir_requires_val_fraction(tmp_path):
    from torchvision.transforms import ToTensor
    from cvic.tunic import _build_loaders
    _make_train_only_dir(tmp_path)
    with pytest.raises(SystemExit):
        _build_loaders(tmp_path, batch_size=4, workers=0, seed=42,
                       train_tf=ToTensor(), val_tf=ToTensor(),
                       training_fraction=0.5, val_fraction=None)


def test_build_loaders_no_val_dir_sum_exceeds_one(tmp_path):
    from torchvision.transforms import ToTensor
    from cvic.tunic import _build_loaders
    _make_train_only_dir(tmp_path)
    with pytest.raises(SystemExit):
        _build_loaders(tmp_path, batch_size=4, workers=0, seed=42,
                       train_tf=ToTensor(), val_tf=ToTensor(),
                       training_fraction=0.7, val_fraction=0.4)


def test_build_loaders_no_val_dir_disjoint(tmp_path):
    from torchvision.transforms import ToTensor
    from cvic.tunic import _build_loaders
    _make_train_only_dir(tmp_path)
    train_loader, val_loader, _ = _build_loaders(
        tmp_path, batch_size=4, workers=0, seed=42,
        train_tf=ToTensor(), val_tf=ToTensor(),
        training_fraction=0.5, val_fraction=0.3,
    )
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    assert train_indices.isdisjoint(val_indices)


def test_build_loaders_no_val_dir_sizes(tmp_path):
    from torchvision.transforms import ToTensor
    from cvic.tunic import _build_loaders
    _make_train_only_dir(tmp_path, n_per_class=20)  # 60 total
    train_loader, val_loader, _ = _build_loaders(
        tmp_path, batch_size=4, workers=0, seed=42,
        train_tf=ToTensor(), val_tf=ToTensor(),
        training_fraction=0.5, val_fraction=0.3,
    )
    # 20*0.5=10 train, 20*0.3=6 val per class → 30 train, 18 val
    assert len(train_loader.dataset) == 30
    assert len(val_loader.dataset) == 18


# ---------------------------------------------------------------------------
# _preflight_check_distribution — no-val-dir imagefolder
# ---------------------------------------------------------------------------

def test_preflight_no_val_dir_requires_val_fraction(tmp_path):
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_dir(tmp_path)
    with pytest.raises(SystemExit):
        _preflight_check_distribution("imagefolder", str(tmp_path), 3, 0.5, None, 42)


def test_preflight_no_val_dir_sum_exceeds_one(tmp_path):
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_dir(tmp_path)
    with pytest.raises(SystemExit):
        _preflight_check_distribution("imagefolder", str(tmp_path), 3, 0.7, 0.4, 42)


def test_preflight_no_val_dir_happy_path(tmp_path, capsys):
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_dir(tmp_path)
    _preflight_check_distribution("imagefolder", str(tmp_path), 3, 0.5, 0.3, 42)
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


# ---------------------------------------------------------------------------
# _preflight_check_distribution — imagefolder stratification regression
# ---------------------------------------------------------------------------

def test_preflight_imagefolder_no_val_dir_stratified(tmp_path, capsys):
    """Sparse-class case (10 classes × 5 samples). Stratified split gives
    4 train + 1 val per class — all classes covered. A non-stratified random
    shuffle on 50 samples drawing 10 for val would leave ~3 classes empty
    (P(empty) ≈ 0.8^5 = 0.328 per class)."""
    from cvic.tunic import _preflight_check_distribution
    classes = tuple(f"c{i:02d}" for i in range(10))
    _make_train_only_dir(tmp_path, classes=classes, n_per_class=5)
    _preflight_check_distribution(
        "imagefolder", str(tmp_path), num_classes=10,
        training_fraction=0.8, val_fraction=0.2, seed=42,
    )
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


# ---------------------------------------------------------------------------
# _preflight_check_distribution — imagefolder with val/ dir
# ---------------------------------------------------------------------------

def _make_train_val_dir(tmp_path, classes=("a", "b", "c"), n_train=20, n_val=10):
    from PIL import Image
    for split, n in [("train", n_train), ("val", n_val)]:
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(
                    np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
                ).save(d / f"{i}.jpg")


def test_preflight_imagefolder_val_dir_happy_path(tmp_path, capsys):
    """train/ and val/ both present, no fractions — validator passes."""
    from cvic.tunic import _preflight_check_distribution
    _make_train_val_dir(tmp_path)
    _preflight_check_distribution(
        "imagefolder", str(tmp_path), num_classes=3,
        training_fraction=1.0, val_fraction=None, seed=42,
    )
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


def test_preflight_imagefolder_val_dir_implicit_full_val(tmp_path, capsys):
    """val_fraction=None must use the entire val/ dir (not error like Path A would)."""
    from cvic.tunic import _preflight_check_distribution
    _make_train_val_dir(tmp_path, n_train=20, n_val=10)
    _preflight_check_distribution(
        "imagefolder", str(tmp_path), num_classes=3,
        training_fraction=1.0, val_fraction=None, seed=42,
    )
    out = capsys.readouterr().out
    # 10 val per class × 3 classes = 30 — must appear in the report.
    assert "30 images" in out or "30," in out


def test_preflight_imagefolder_val_dir_with_training_fraction(tmp_path, capsys):
    """training_fraction < 1.0 subsamples training data; validator passes."""
    from cvic.tunic import _preflight_check_distribution
    _make_train_val_dir(tmp_path, n_train=20, n_val=10)
    _preflight_check_distribution(
        "imagefolder", str(tmp_path), num_classes=3,
        training_fraction=0.5, val_fraction=None, seed=42,
    )
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


def test_preflight_imagefolder_val_dir_with_val_fraction(tmp_path, capsys):
    """val_fraction < 1.0 subsamples val data; validator passes."""
    from cvic.tunic import _preflight_check_distribution
    _make_train_val_dir(tmp_path, n_train=20, n_val=10)
    _preflight_check_distribution(
        "imagefolder", str(tmp_path), num_classes=3,
        training_fraction=1.0, val_fraction=0.5, seed=42,
    )
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


# ---------------------------------------------------------------------------
# _preflight_check_distribution — no-val-split webdataset
# ---------------------------------------------------------------------------

def _make_train_only_wds(tmp_path, classes=tuple(f"c{i:02d}" for i in range(10)),
                          n_per_class=5):
    """Create a minimal WDS train-only dataset (PNG + .cls samples, single shard)."""
    import io
    import tarfile
    from PIL import Image

    wds_dir = tmp_path / "wds"
    train_dir = wds_dir / "train"
    train_dir.mkdir(parents=True)

    samples = []
    for cls in classes:
        for i in range(n_per_class):
            buf = io.BytesIO()
            Image.fromarray(
                np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            ).save(buf, format="PNG")
            samples.append((buf.getvalue(), cls))

    with tarfile.open(train_dir / "shard-000000.tar", "w") as tf:
        for i, (png_bytes, cls) in enumerate(samples):
            key = f"000000_{i:06d}"
            for ext, data in [(".png", png_bytes), (".cls", cls.encode())]:
                info = tarfile.TarInfo(name=f"{key}{ext}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    with open(wds_dir / "dataset_info.json", "w") as f:
        json.dump({
            "format": "webdataset",
            "classes": list(classes),
            "class_names": {c: c for c in classes},
            "splits": {"train": {"num_shards": 1, "num_samples": len(samples)}},
        }, f)


def test_preflight_wds_no_val_split_requires_val_fraction(tmp_path):
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_wds(tmp_path)
    with pytest.raises(SystemExit):
        _preflight_check_distribution("webdataset", str(tmp_path), 10, 0.5, None, 42)


def test_preflight_wds_no_val_split_sum_exceeds_one(tmp_path):
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_wds(tmp_path)
    with pytest.raises(SystemExit):
        _preflight_check_distribution("webdataset", str(tmp_path), 10, 0.7, 0.4, 42)


def test_preflight_wds_no_val_split_stratified(tmp_path, capsys):
    """Sparse-class case (10 classes × 5 samples) that a non-stratified
    random shuffle would flunk: P(any class has 0 val) is ~0.97 across seeds.
    Stratified split gives 4 train + 1 val per class — all classes covered."""
    from cvic.tunic import _preflight_check_distribution
    _make_train_only_wds(tmp_path, n_per_class=5)
    _preflight_check_distribution(
        "webdataset", str(tmp_path), num_classes=10,
        training_fraction=0.8, val_fraction=0.2, seed=42,
    )
    out = capsys.readouterr().out
    assert "Training set" in out
    assert "Validation set" in out


# ---------------------------------------------------------------------------
# --shuffle split behavior
# ---------------------------------------------------------------------------

def test_shuffle_split_behavior(tmp_path):
    """Without shuffle all trials share the same split; with shuffle each gets a unique one."""
    from torchvision import datasets
    from cvic.common_cvic import make_stratified_split
    from PIL import Image

    for cls in ["a", "b", "c"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(20):
            Image.fromarray(
                np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            ).save(d / f"{i}.jpg")
    ds = datasets.ImageFolder(str(tmp_path))

    base_seed = 42
    shuffle_seed = 100

    # No shuffle: both trials derive split from the same base_seed
    _, val_t1 = make_stratified_split(ds, seed=base_seed)
    _, val_t2 = make_stratified_split(ds, seed=base_seed)
    assert val_t1.indices == val_t2.indices

    # With shuffle: trial 1 uses shuffle_seed+1, trial 2 uses shuffle_seed+2
    _, val_s1 = make_stratified_split(ds, seed=shuffle_seed + 1)
    _, val_s2 = make_stratified_split(ds, seed=shuffle_seed + 2)
    assert val_s1.indices != val_s2.indices


# ---------------------------------------------------------------------------
# smoke test (CPU, end-to-end)
# ---------------------------------------------------------------------------

def test_smoke(tmp_path):
    """End-to-end smoke test: tuning + final training on synthetic data."""
    import argparse
    from PIL import Image
    from cvic.tunic import run_smoke_test

    for split in ["train", "val"]:
        for cls in ["cat", "dog", "bird"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(10):
                Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(device="cpu", smoke_test=True)
    run_smoke_test(args)


# ---------------------------------------------------------------------------
# Tuning reproducibility
# ---------------------------------------------------------------------------

def test_tuning_reproducible(tmp_path):
    """Run tuning twice with same seed, verify identical results (reproducibility test)."""
    import argparse
    from PIL import Image
    from cvic.tunic import run_tuning

    # Create synthetic data: 2 classes, 8 images each per split
    for split in ["train", "val"]:
        for cls in ["a", "b"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(8):
                Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                ).save(d / f"img_{i}.jpg")

    # Run 1: tuning with seed=42
    results1_path = tmp_path / "run1.json"
    args1 = argparse.Namespace(
        data=str(tmp_path),
        model="resnet18",
        pretrained=False,
        n_trials=3,
        epochs=1,
        batch_size=4,
        seed=42,
        device="cpu",
        workers=0,
        img_size=64,
        freeze_backbone=0,
        training_fraction=1.0,
        val_fraction=1.0,
        prefix=str(results1_path.with_suffix("")),
        resume=None,
        search_space=None,
        final=None,
        smoke_test=False,
        num_train_workers=1,
        ray_address=None,
        ray_storage=None,
        tune_metric="val_auroc",
        amp=False,
        shuffle=None,
        test_data=None,
        sampler="tpe",
        scheduler="asha",
        optuna_storage=None,
    )
    run_tuning(args1)

    # Run 2: tuning with same seed=42
    results2_path = tmp_path / "run2.json"
    args2 = argparse.Namespace(**vars(args1))
    args2.prefix = str(results2_path.with_suffix(""))

    run_tuning(args2)

    # Load and compare results
    with open(results1_path) as f:
        r1 = json.load(f)
    with open(results2_path) as f:
        r2 = json.load(f)

    # Check best params are identical
    assert r1["best_params"] == r2["best_params"], "Best params differ between runs"
    # Check best metrics are identical
    assert r1["best_val_acc"] == r2["best_val_acc"], "best_val_acc differs"
    assert r1["best_val_auroc"] == r2["best_val_auroc"], "best_val_auroc differs"
    # Check number of completed trials matches
    assert r1["completed_trials"] == r2["completed_trials"], "Completed trial count differs"


# ---------------------------------------------------------------------------
# HP-importance pathway: random sampler + no scheduler + sqlite storage
# ---------------------------------------------------------------------------

def test_random_sampler_no_scheduler_with_sqlite(tmp_path):
    """Tuning with --sampler random --scheduler none --optuna-storage sqlite://...

    Validates the wiring that lets manuscript-grade HP-importance studies run:
      - random sampler uniformly covers the search space (needed for fANOVA);
      - ASHA disabled so every trial runs to full --epochs (no early-stop bias);
      - study DB persists so optuna.importance can be called post-hoc.
    """
    import argparse
    from PIL import Image
    from cvic.tunic import run_tuning
    import optuna

    for split in ["train", "val"]:
        for cls in ["a", "b"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(8):
                Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                ).save(d / f"img_{i}.jpg")

    db_path = tmp_path / "study.db"
    results_path = tmp_path / "imp.json"
    n_trials = 6
    args = argparse.Namespace(
        data=str(tmp_path),
        model="resnet18",
        pretrained=False,
        n_trials=n_trials,
        epochs=1,
        batch_size=4,
        seed=7,
        device="cpu",
        workers=0,
        img_size=64,
        freeze_backbone=0,
        training_fraction=1.0,
        val_fraction=1.0,
        prefix=str(results_path.with_suffix("")),
        resume=None,
        search_space=None,
        final=None,
        smoke_test=False,
        num_train_workers=1,
        ray_address=None,
        ray_storage=None,
        tune_metric="val_auroc",
        amp=False,
        shuffle=None,
        test_data=None,
        sampler="random",
        scheduler="none",
        optuna_storage=f"sqlite:///{db_path}",
    )
    run_tuning(args)

    assert db_path.exists(), "sqlite study DB was not created"
    study = optuna.load_study(study_name=args.prefix, storage=f"sqlite:///{db_path}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == n_trials, (
        f"Expected {n_trials} completed trials in study DB, got {len(completed)}"
    )

    # fANOVA needs the study to be persisted and the parameter space recoverable.
    importances = optuna.importance.get_param_importances(study)
    assert isinstance(importances, dict) and len(importances) > 0, (
        "optuna.importance.get_param_importances returned no entries"
    )


# ---------------------------------------------------------------------------
# _stratified_disjoint_split — train/val overlap fix for WebDataset no-val case
# ---------------------------------------------------------------------------

def test_stratified_disjoint_split_no_overlap():
    """Train and val keys must never overlap."""
    from cvic.tunic import _stratified_disjoint_split
    samples = [(f"s{i:04d}", i % 2) for i in range(100)]
    train_keys, val_keys, dist = _stratified_disjoint_split(samples, 0.8, 0.2, seed=42)
    assert len(train_keys & val_keys) == 0


def test_stratified_disjoint_split_class_balanced():
    """Each class should be split per the requested fractions."""
    from cvic.tunic import _stratified_disjoint_split
    # 60 class-0 + 40 class-1
    samples = [(f"a{i:04d}", 0) for i in range(60)] + [(f"b{i:04d}", 1) for i in range(40)]
    train_keys, val_keys, dist = _stratified_disjoint_split(samples, 0.8, 0.2, seed=0)
    assert dist[0]["train"] == 48 and dist[0]["val"] == 12
    assert dist[1]["train"] == 32 and dist[1]["val"] == 8
    # Check that both classes are present in train and val
    train_labels = {label for k, label in samples if k in train_keys}
    val_labels = {label for k, label in samples if k in val_keys}
    assert train_labels == {0, 1}
    assert val_labels == {0, 1}


def test_stratified_disjoint_split_reproducible():
    """Same seed → identical splits."""
    from cvic.tunic import _stratified_disjoint_split
    samples = [(f"s{i:04d}", i % 3) for i in range(90)]
    t1, v1, _ = _stratified_disjoint_split(samples, 0.7, 0.3, seed=123)
    t2, v2, _ = _stratified_disjoint_split(samples, 0.7, 0.3, seed=123)
    assert t1 == t2 and v1 == v2


def test_stratified_disjoint_split_different_seeds_differ():
    """Different seeds → different splits."""
    from cvic.tunic import _stratified_disjoint_split
    samples = [(f"s{i:04d}", i % 3) for i in range(90)]
    t1, _, _ = _stratified_disjoint_split(samples, 0.7, 0.3, seed=1)
    t2, _, _ = _stratified_disjoint_split(samples, 0.7, 0.3, seed=2)
    assert t1 != t2
