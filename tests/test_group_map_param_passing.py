"""Ray-free proof that trials receive group_map via their PARAMETER, not config.

Option 2 of the log-bloat fix moved group_map out of the Ray param_space/config
(where Ray pretty-printed all ~10k entries into every trial log) and into
tune.with_parameters, which injects it as a trainable keyword argument. Ray runs
trials in separate worker processes, so the end-to-end run_cv / run_tuning tests
can only prove "completes" — NOT that the map still reaches the splitter, which is
exactly this refactor's failure mode (a silent group_map=None would still finish).

These tests call the trial functions directly, with the splitter stubbed to abort
before training, proving:
  * the trial forwards its group_map PARAMETER to the splitter, and
  * it does NOT read group_map from config (a stale config value is ignored).

Combined with the unchanged _build_loaders / _make_folds disjointness tests
(test_group_map_wiring.py, test_cv_folds.py), this covers the full chain.
"""

import csv

import numpy as np
import pytest
from PIL import Image

from cvic.common_cvic import _load_group_map


class _Stop(Exception):
    """Raised by a stubbed splitter to abort the trial before any training runs."""


class _FakeRemote:
    def remote(self):
        return "token"


class _FakeCounter:
    next = _FakeRemote()


def _patch_ray_counter(monkeypatch):
    """Let the trial's ray.get_actor('trial_counter') / ray.get(...) work with no
    live Ray cluster."""
    import ray
    monkeypatch.setattr(ray, "get_actor", lambda name: _FakeCounter())
    monkeypatch.setattr(ray, "get", lambda token: 1)


def _base_hp_config():
    """Every hyperparameter key the trials read before splitting. Carries a bogus
    group_map to prove the trial ignores config as a source for it."""
    return dict(
        device="cpu", n_trials=1, seed=0, epochs=1,
        lr=1e-3, weight_decay=1e-4, label_smoothing=0.0, drop_rate=0.0,
        randaugment_magnitude=0, randaugment_num_ops=1,
        mixup_alpha=0.0, cutmix_alpha=0.0, optimizer="AdamW",
        num_classes=2, img_size=32, batch_size=4, dataloader_workers=0,
        group_map="CONFIG_VALUE_MUST_BE_IGNORED",
    )


def _make_grouped_imagefolder(root, n_lesions_per_class=4, imgs_per_lesion=2):
    """train/<class>/<lesion>_<k>.jpg sharing lesion ids; writes a manifest CSV."""
    mapping = {}
    for cls in ("benign", "malignant"):
        d = root / "train" / cls
        d.mkdir(parents=True)
        for les in range(n_lesions_per_class):
            lesion_id = f"{cls[:3]}_les{les:03d}"
            for k in range(imgs_per_lesion):
                stem = f"{lesion_id}_{k}"
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(d / f"{stem}.jpg")
                mapping[stem] = lesion_id
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "lesion_id"])
        for stem, les in mapping.items():
            w.writerow([stem, les])
    return mapping, manifest


# ---------------------------------------------------------------------------
# tunic: _tune_trial forwards its group_map parameter to _build_loaders
# ---------------------------------------------------------------------------

def test_tune_trial_uses_group_map_parameter_not_config(monkeypatch):
    from cvic import tunic

    captured = {}

    def _stub_build_loaders(*args, **kwargs):
        captured["group_map"] = kwargs.get("group_map")
        raise _Stop

    monkeypatch.setattr(tunic, "_build_loaders", _stub_build_loaders)
    _patch_ray_counter(monkeypatch)

    gm = {"a": "lesion1", "b": "lesion1", "c": "lesion2"}
    config = _base_hp_config()
    config.update(data="/does/not/matter", training_fraction=0.7, val_fraction=0.3)

    with pytest.raises(_Stop):
        tunic._tune_trial(config, group_map=gm)

    assert captured["group_map"] is gm, \
        "trial did not forward its group_map parameter to _build_loaders"
    assert captured["group_map"] != config["group_map"], \
        "trial used the (ignored) config group_map value"


def test_tune_trial_ungrouped_passes_none(monkeypatch):
    """No group_map parameter -> splitter gets None (byte-identical ungrouped path)."""
    from cvic import tunic

    captured = {}

    def _stub_build_loaders(*args, **kwargs):
        captured["group_map"] = kwargs.get("group_map")
        raise _Stop

    monkeypatch.setattr(tunic, "_build_loaders", _stub_build_loaders)
    _patch_ray_counter(monkeypatch)

    config = _base_hp_config()
    config.update(data="/does/not/matter", training_fraction=0.7, val_fraction=0.3)

    with pytest.raises(_Stop):
        tunic._tune_trial(config)  # no group_map kwarg

    assert captured["group_map"] is None


# ---------------------------------------------------------------------------
# cvic: _cvic_trial forwards its group_map parameter into groups -> _make_folds
# ---------------------------------------------------------------------------

def test_cvic_trial_uses_group_map_parameter_not_config(monkeypatch, tmp_path):
    from cvic import cvic as cvic_mod

    mapping, manifest = _make_grouped_imagefolder(tmp_path)
    gm = _load_group_map(str(manifest))

    captured = {}

    def _stub_make_folds(n_samples, labels, n_folds, stratified, seed, groups=None):
        captured["groups"] = groups
        raise _Stop

    monkeypatch.setattr(cvic_mod, "_make_folds", _stub_make_folds)
    _patch_ray_counter(monkeypatch)

    config = _base_hp_config()
    config.update(
        data=str(tmp_path), n_folds=2, n_repeats=1, stratified=True,
        pooling=False, tune_metric="val_acc", use_amp=False, freeze_backbone=0,
    )

    with pytest.raises(_Stop):
        cvic_mod._cvic_trial(config, group_map=gm)

    groups = captured["groups"]
    assert groups is not None, \
        "trial did not forward group_map -> groups into _make_folds"
    assert len(groups) == len(mapping), "groups length mismatch vs number of samples"
    assert set(groups) == set(mapping.values()), "groups do not reflect the lesion map"


def test_cvic_trial_ungrouped_passes_none_groups(monkeypatch, tmp_path):
    """No group_map parameter -> _make_folds gets groups=None (ungrouped path)."""
    from cvic import cvic as cvic_mod

    _make_grouped_imagefolder(tmp_path)

    captured = {}

    def _stub_make_folds(n_samples, labels, n_folds, stratified, seed, groups=None):
        captured["groups"] = groups
        raise _Stop

    monkeypatch.setattr(cvic_mod, "_make_folds", _stub_make_folds)
    _patch_ray_counter(monkeypatch)

    config = _base_hp_config()
    config.update(
        data=str(tmp_path), n_folds=2, n_repeats=1, stratified=True,
        pooling=False, tune_metric="val_acc", use_amp=False, freeze_backbone=0,
    )

    with pytest.raises(_Stop):
        cvic_mod._cvic_trial(config)  # no group_map kwarg

    assert captured["groups"] is None
