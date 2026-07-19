"""Wiring tests for --group-map in cvic (cross-validation).

The group-disjoint fold guarantee itself is unit-tested in test_cv_folds.py
(the grouped _make_folds cases). These tests prove the CLI plumbing delivers the
map end-to-end: parse_args -> run_cv loads it -> trial config -> _cvic_trial
builds per-sample groups -> _make_folds(groups=...).
"""

import argparse
import csv
import sys

import numpy as np
import pytest
from PIL import Image

from cvic.cvic import parse_args, run_cv


def _make_grouped_imagefolder(root, n_lesions_per_class=8, imgs_per_lesion=2):
    """train/<class>/<lesion>_<k>.jpg with several images sharing a lesion id;
    also writes a manifest CSV (image_id,lesion_id)."""
    mapping = {}
    for cls in ("benign", "malignant"):
        d = root / "train" / cls
        d.mkdir(parents=True)
        for les in range(n_lesions_per_class):
            lesion_id = f"{cls[:3]}_les{les:03d}"
            for k in range(imgs_per_lesion):
                stem = f"{lesion_id}_{k}"
                Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                ).save(d / f"{stem}.jpg")
                mapping[stem] = lesion_id
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "lesion_id"])
        for stem, les in mapping.items():
            w.writerow([stem, les])
    return manifest


def test_cli_group_map(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["cvic", "--data", "/tmp", "--model", "resnet18",
                         "--group-map", "/tmp/manifest.csv"])
    assert parse_args().group_map == "/tmp/manifest.csv"


def test_run_cv_with_group_map_end_to_end(tmp_path):
    """Full CV search with --group-map completes and writes results — proving the
    map reaches _make_folds through Ray without error."""
    manifest = _make_grouped_imagefolder(tmp_path)

    args = argparse.Namespace(
        data=str(tmp_path), model="resnet18", pretrained=False,
        n_trials=1, epochs=1, folds=2, repeats=1,
        stratified=True, pooling=False, tune_metric="val_acc",
        batch_size=4, prefix=str(tmp_path / "cvic_grouped"),
        seed=0, device="cpu", cpu=False, repro=False, workers=0,
        img_size=32, freeze_backbone=0, amp=False, search_space=None,
        ray_address=None, ray_storage=str(tmp_path / "ray_results"),
        test_data=None, no_final=True, checkpoint=False, resume=None,
        smoke_test=False, random_seed=False,
        group_map=str(manifest),
    )
    run_cv(args)

    import json
    with open(f"{args.prefix}.json") as f:
        result = json.load(f)
    assert result["n_folds"] == 2
    assert result["completed_trials"] == 1
    assert result["best_val_acc"] is not None
