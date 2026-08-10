"""Unit and smoke tests for cvic."""

import argparse
import math
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _compute_metric
# ---------------------------------------------------------------------------

def test_compute_metric_auroc_perfect():
    from cvic.cvic import _compute_metric
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    assert _compute_metric(probs, labels, "val_auroc") == 1.0


def test_compute_metric_acc_perfect():
    from cvic.cvic import _compute_metric
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    labels = np.array([0, 1, 0])
    assert _compute_metric(probs, labels, "val_acc") == 1.0


def test_compute_metric_acc_partial():
    from cvic.cvic import _compute_metric
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 1, 0])  # second wrong, third wrong
    acc = _compute_metric(probs, labels, "val_acc")
    assert acc == pytest.approx(1 / 3)


def test_compute_metric_unknown_falls_back_to_auroc():
    from cvic.cvic import _compute_metric
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    result = _compute_metric(probs, labels, "something_else")
    assert result == 1.0


def test_uses_ray_gpu_auto_is_false_without_cuda(monkeypatch):
    import cvic.common_cvic as common
    from cvic.cvic import _uses_ray_gpu

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(common.torch.backends.mps, "is_available", lambda: False)

    assert _uses_ray_gpu("auto") is False
    assert _uses_ray_gpu("cpu") is False
    assert _uses_ray_gpu("cuda") is True


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_cli_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cvic", "--data", "/tmp", "--model", "resnet18"])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.folds == 5
    assert args.repeats == 1
    assert args.stratified is True
    assert args.pooling is False
    assert args.tune_metric == "val_auroc"
    assert args.n_trials == 30
    assert args.epochs == 30


def test_cli_folds_repeats(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18",
        "--folds", "3", "--repeats", "2",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.folds == 3
    assert args.repeats == 2


def test_cli_pooling(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--pooling",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.pooling is True


def test_cli_no_stratified(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--no-stratified",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.stratified is False


def test_cli_tune_metric(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--tune-metric", "val_acc",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.tune_metric == "val_acc"


def test_cli_repro(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--repro",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.repro is True


def test_cli_cpu(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--cpu",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.cpu is True


def test_cli_prefix(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--prefix", "myexp",
    ])
    from cvic.cvic import parse_args
    args = parse_args()
    assert args.prefix == "myexp"


# ---------------------------------------------------------------------------
# smoke test (CPU, end-to-end with Ray local cluster)
# ---------------------------------------------------------------------------

def test_smoke_cvic(tmp_path):
    """End-to-end smoke test: CV search on synthetic data, CPU, 1 trial, 2 folds, 1 epoch."""
    from PIL import Image
    from cvic.cvic import run_cv

    for cls in ["cat", "dog", "bird"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(12):
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(
        data=str(tmp_path),
        model="resnet18",
        pretrained=False,
        n_trials=1,
        epochs=1,
        folds=2,
        repeats=1,
        stratified=True,
        pooling=False,
        tune_metric="val_acc",
        batch_size=4,
        prefix=str(tmp_path / "cvic_smoke"),
        seed=0,
        device="cpu",
        cpu=False,
        repro=False,
        workers=0,
        img_size=32,
        freeze_backbone=0,
        amp=False,
        search_space=None,
        ray_address=None,
        ray_storage=str(tmp_path / "ray_results"),
        test_data=None,
        no_final=True,
        checkpoint=False,
        resume=None,
        smoke_test=False,
        random_seed=False,
    )

    run_cv(args)

    import json
    out_path = f"{args.prefix}.json"
    with open(out_path) as f:
        result = json.load(f)
    assert "best_val_acc" in result
    assert result["n_folds"] == 2
    assert result["completed_trials"] == 1
    assert result["best_params"]["batch_size"] == 4


@pytest.mark.parametrize("pooling", [False, True])
def test_val_acc_reported_when_tuning_auroc(tmp_path, pooling):
    """Regression: tuning on val_auroc must still report a real val_acc, not nan.

    _cvic_trial used to compute only the tune_metric and stub the other column
    with float('nan'). With the default --tune-metric val_auroc that made every
    trial's val_acc nan. Both metrics are now computed from the OOF predictions,
    so best_val_acc must be a finite number in [0, 1] even though selection is
    done on AUROC. Covers both the per-fold (pooling=False) and pooled
    (pooling=True) aggregation paths.
    """
    from PIL import Image
    from cvic.cvic import run_cv
    import json

    for cls in ["cat", "dog", "bird"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(12):
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(
        data=str(tmp_path),
        model="resnet18",
        pretrained=False,
        n_trials=1,
        epochs=1,
        folds=2,
        repeats=1,
        stratified=True,
        pooling=pooling,
        tune_metric="val_auroc",  # the default that triggered the nan bug
        batch_size=4,
        prefix=str(tmp_path / "cvic_valacc"),
        seed=0,
        device="cpu",
        cpu=False,
        repro=False,
        workers=0,
        img_size=32,
        freeze_backbone=0,
        amp=False,
        search_space=None,
        ray_address=None,
        ray_storage=str(tmp_path / "ray_results"),
        test_data=None,
        no_final=True,
        checkpoint=False,
        resume=None,
        smoke_test=False,
        random_seed=False,
    )

    run_cv(args)

    with open(f"{args.prefix}.json") as f:
        result = json.load(f)

    # Selection metric is still AUROC and present.
    assert result["best_val_auroc"] is not None
    assert not math.isnan(result["best_val_auroc"])
    # The previously-stubbed metric is now a genuine, finite accuracy.
    acc = result["best_val_acc"]
    assert acc is not None
    assert not math.isnan(acc), "val_acc should be computed, not nan, when tuning on AUROC"
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# --repeats: validation + mechanism + end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["0", "-1", "-5"])
def test_cli_repeats_rejects_nonpositive(monkeypatch, bad):
    """--repeats must be a positive integer; 0 and negatives are rejected at parse."""
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--repeats", bad,
    ])
    from cvic.cvic import parse_args
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("bad", ["2.5", "abc", "1e3"])
def test_cli_repeats_rejects_noninteger(monkeypatch, bad):
    """Non-integer --repeats values are rejected at parse."""
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--repeats", bad,
    ])
    from cvic.cvic import parse_args
    with pytest.raises(SystemExit):
        parse_args()


def test_positive_int_type():
    """The argparse type accepts >= 1 and rejects 0, negatives, and non-integers."""
    import argparse
    from cvic.cvic import _positive_int
    assert _positive_int("1") == 1
    assert _positive_int("10") == 10
    for bad in ["0", "-1", "2.5", "abc"]:
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)


def test_run_cv_rejects_nonpositive_repeats(tmp_path):
    """run_cv (called programmatically with a Namespace) also guards --repeats,
    exiting before any Ray/dataset work rather than running an empty repeat loop."""
    from cvic.cvic import run_cv
    args = argparse.Namespace(
        data=str(tmp_path), model="resnet18", pretrained=False,
        n_trials=1, epochs=1, folds=2, repeats=0, stratified=True, pooling=False,
        tune_metric="val_acc", batch_size=4, prefix=str(tmp_path / "cvic_bad"),
        seed=0, device="cpu", cpu=False, repro=False, workers=0, img_size=32,
        freeze_backbone=0, amp=False, search_space=None, ray_address=None,
        ray_storage=str(tmp_path / "ray_results"), test_data=None, no_final=True,
        checkpoint=False, resume=None, smoke_test=False, random_seed=False,
    )
    with pytest.raises(SystemExit):
        run_cv(args)


def test_make_folds_repeats_are_distinct_valid_partitions():
    """Each repeat uses a different seed (base + repeat*10000) and must still yield
    a valid partition (folds disjoint, covering every sample); across repeats the
    fold assignments genuinely differ, which is what makes repeated CV meaningful."""
    from cvic.cvic import _make_folds
    N, n_folds = 40, 4
    labels = [0] * 20 + [1] * 20
    base_seed = 42

    assignments = []
    for repeat in range(3):
        seed = base_seed + repeat * 10000  # mirrors run_cv's repeat_seed
        splits = _make_folds(N, labels, n_folds, stratified=True, seed=seed)
        # Valid partition: val sets are disjoint and cover all N exactly once.
        assign = np.full(N, -1)
        for fold_idx, (_, val_idx) in enumerate(splits):
            assert np.all(assign[val_idx] == -1), "sample assigned to two val folds"
            assign[val_idx] = fold_idx
        assert np.all(assign >= 0), "some sample never appeared in a val fold"
        assignments.append(assign)

    # Different seeds → different partitions (not all repeats identical).
    assert not np.array_equal(assignments[0], assignments[1])
    assert not np.array_equal(assignments[1], assignments[2])


# ---------------------------------------------------------------------------
# --folds: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["0", "-1"])
def test_cli_folds_rejects_nonpositive(monkeypatch, bad):
    """--folds must be a positive integer; 0 and negatives are rejected at parse."""
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--folds", bad,
    ])
    from cvic.cvic import parse_args
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("bad", ["2.5", "abc"])
def test_cli_folds_rejects_noninteger(monkeypatch, bad):
    """Non-integer --folds values are rejected at parse."""
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--folds", bad,
    ])
    from cvic.cvic import parse_args
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("folds", [0, 1])
def test_run_cv_rejects_too_few_folds(tmp_path, folds):
    """run_cv guards --folds: 0 (also caught at parse) and 1 (parses, but CV needs
    >= 2) exit before any Ray/dataset work rather than crashing later in sklearn."""
    from cvic.cvic import run_cv
    args = argparse.Namespace(
        data=str(tmp_path), model="resnet18", pretrained=False,
        n_trials=1, epochs=1, folds=folds, repeats=1, stratified=True, pooling=False,
        tune_metric="val_acc", batch_size=4, prefix=str(tmp_path / "cvic_badfolds"),
        seed=0, device="cpu", cpu=False, repro=False, workers=0, img_size=32,
        freeze_backbone=0, amp=False, search_space=None, ray_address=None,
        ray_storage=str(tmp_path / "ray_results"), test_data=None, no_final=True,
        checkpoint=False, resume=None, smoke_test=False, random_seed=False,
    )
    with pytest.raises(SystemExit):
        run_cv(args)


@pytest.mark.parametrize("pooling", [False, True])
def test_repeats_end_to_end(tmp_path, pooling):
    """End-to-end: --repeats 2 runs the full repeated CV and reports finite metrics.

    Covers both aggregation paths: pooling=True averages the out-of-fold
    probabilities across repeats (probs_accum / n_repeats); pooling=False averages
    the per-fold metrics across all repeat*fold folds.
    """
    from PIL import Image
    from cvic.cvic import run_cv
    import json

    for cls in ["cat", "dog", "bird"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(12):
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(
        data=str(tmp_path), model="resnet18", pretrained=False,
        n_trials=1, epochs=1, folds=2, repeats=2, stratified=True, pooling=pooling,
        tune_metric="val_acc", batch_size=4, prefix=str(tmp_path / "cvic_repeats"),
        seed=0, device="cpu", cpu=False, repro=False, workers=0, img_size=32,
        freeze_backbone=0, amp=False, search_space=None, ray_address=None,
        ray_storage=str(tmp_path / "ray_results"), test_data=None, no_final=True,
        checkpoint=False, resume=None, smoke_test=False, random_seed=False,
    )

    run_cv(args)

    with open(f"{args.prefix}.json") as f:
        result = json.load(f)

    assert result["n_repeats"] == 2
    assert result["n_folds"] == 2
    assert result["completed_trials"] == 1
    acc = result["best_val_acc"]
    assert acc is not None and not math.isnan(acc)
    assert 0.0 <= acc <= 1.0


def test_repro_deterministic(tmp_path):
    """Verify --repro flag produces identical results across runs."""
    from PIL import Image
    from cvic.cvic import run_cv
    import json

    # Create synthetic dataset
    for cls in ["cat", "dog"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(16):
            Image.fromarray(
                np.random.RandomState(42).randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(d / f"img_{i}.jpg")

    # Run twice with --repro
    results = []
    for run_id in range(2):
        args = argparse.Namespace(
            data=str(tmp_path),
            model="resnet18",
            pretrained=False,
            n_trials=1,
            epochs=1,
            folds=2,
            repeats=1,
            stratified=True,
            pooling=True,  # pool to get single metric per run
            tune_metric="val_acc",
            batch_size=4,
            prefix=str(tmp_path / f"cvic_repro_run{run_id}"),
            seed=42,
            device="cpu",
            cpu=False,
            repro=True,  # Enable reproducibility mode
            workers=0,
            img_size=32,
            freeze_backbone=0,
            amp=False,
            search_space=None,
            ray_address=None,
            ray_storage=str(tmp_path / f"ray_results_run{run_id}"),
            test_data=None,
            no_final=True,
            checkpoint=False,
            resume=None,
            smoke_test=False,
            random_seed=False,
        )
        run_cv(args)
        with open(f"{args.prefix}.json") as f:
            results.append(json.load(f))

    # With --repro, best_val_acc should be identical
    assert results[0]["best_val_acc"] == results[1]["best_val_acc"]
