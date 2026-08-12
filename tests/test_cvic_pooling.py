"""Ray-free proof of the development-AUROC estimator computed by _cvic_trial.

cvic reports the CV "development AUROC" two ways, chosen by --pooling:

  * default (pooling=False): mean of the per-fold AUROCs over all R x k folds,
    and NO across-repeat spread (val_auroc_sd is nan).
  * --pooling (pooling=True): for each repeat, pool its folds into ONE full-dev
    out-of-fold AUROC (single-fold models, no cross-repeat ensembling), then
    average those per-repeat AUROCs; val_auroc_sd is their spread across repeats.

The old --pooling averaged probabilities across repeats and scored once — an
ensemble that inflates AUROC and carries no sd. These tests pin the new contract
at the tune.report boundary so a regression to the ensemble form (or a dropped
sd) fails loudly.

Ray runs trials in worker processes, so an end-to-end run can only prove
"completes". Here we call _cvic_trial directly with training stubbed and a
deterministic fake model, capturing what it reports.

Key simplification: a repeat's folds partition the dev set exactly once, so the
per-repeat POOLED-over-folds AUROC depends only on that repeat's per-sample
predictions, NOT on how the folds were cut. The fake model makes each sample's
prediction a deterministic function of (sample-index, repeat-index), so the test
reconstructs the expected per-repeat AUROCs — and hence their mean and sd —
without knowing the fold assignments at all.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

# --- fake-model signal: deterministic per (sample index i, repeat r) -----------
# A small label term (so classes are separable-ish) plus a sample/repeat-dependent
# perturbation that reshuffles ranks between repeats -> per-repeat AUROC varies
# -> a real, non-zero across-repeat sd, and mean-of-repeats != ensemble.
_N0 = 10          # class-0 samples
_N1 = 10          # class-1 samples
_N = _N0 + _N1
_LABELS = np.array([0] * _N0 + [1] * _N1)   # ImageFolder order: c0 block then c1


def _score(i, r):
    """Class-1 logit for sample i in repeat r (class-0 logit is fixed at 0)."""
    y = _LABELS[i]
    return 0.5 * (2 * y - 1) + 0.6 * math.cos(i + 1.3 * r)


class _FakeModel(nn.Module):
    """No parameters. Decodes each sample's index from its constant pixel fill
    (value == dataset index, preserved by the identity transform below) and emits
    logits [0, _score(i, repeat)]. `repeat` is supplied out-of-band by the training
    stub, which cvic calls once per fold in fold order."""

    def __init__(self, state):
        super().__init__()
        self._state = state

    def forward(self, x):                      # x: (B, 3, H, W), constant per sample
        idx = torch.round(x[:, 0].mean(dim=(1, 2)) * 255.0).long()
        r = self._state["repeat"]
        s = torch.tensor([_score(int(i), r) for i in idx], dtype=torch.float32)
        return torch.stack([torch.zeros_like(s), s], dim=1)


def _make_imagefolder(root):
    """Two classes; each image is a constant fill equal to its eventual dataset
    index, so the fake model can recover the index (and thus the label/repeat
    signal) from pixels. Filenames keep sort order == index order within a class."""
    for cls, lo in (("c0", 0), ("c1", _N0)):
        d = root / "train" / cls
        d.mkdir(parents=True)
        for k in range(_N0 if cls == "c0" else _N1):
            i = lo + k
            Image.new("RGB", (8, 8), (i, i, i)).save(d / f"{i:03d}.png")


class _Rec:
    """Captures the single tune.report(...) payload."""
    def __init__(self):
        self.metrics = None


def _run_trial(monkeypatch, tmp_path, *, pooling, n_folds=5, n_repeats=4):
    from cvic import cvic as cvic_mod
    from cvic import common_cvic as ck
    import ray
    from ray import tune

    _make_imagefolder(tmp_path)
    state = {"calls": -1, "repeat": 0, "n_folds": n_folds}

    # cvic pulls create_model / train_one_epoch / build_transforms etc. off the
    # common_cvic module (ck) at call time, so patch them there.
    #
    # Identity-ish transform: constant PIL fill v -> tensor value v/255 exactly,
    # so the model's index decode is exact (no ImageNet normalization to invert).
    monkeypatch.setattr(ck, "build_transforms",
                        lambda *a, **k: T.Compose([T.Resize((8, 8)), T.ToTensor()]))
    # No real training; advance the repeat counter (cvic calls this once per fold,
    # in repeat-major fold order, before that fold's eval).
    def _fake_train(*a, **k):
        state["calls"] += 1
        state["repeat"] = state["calls"] // state["n_folds"]
    monkeypatch.setattr(ck, "train_one_epoch", _fake_train)
    monkeypatch.setattr(ck, "create_model", lambda *a, **k: _FakeModel(state))
    monkeypatch.setattr(ck, "get_optimizer", lambda *a, **k: None)
    monkeypatch.setattr(ck, "build_scheduler", lambda *a, **k: None)

    # Ray counter (no live cluster).
    class _R:  # noqa: D401
        def remote(self_):
            return "tok"
    class _C:
        next = _R()
    monkeypatch.setattr(ray, "get_actor", lambda name: _C())
    monkeypatch.setattr(ray, "get", lambda tok: 1)

    rec = _Rec()
    monkeypatch.setattr(tune, "report", lambda metrics: setattr(rec, "metrics", metrics))

    config = dict(
        data=str(tmp_path), model="fake", pretrained=False,
        device="cpu", n_trials=1, seed=0, epochs=1,
        n_folds=n_folds, n_repeats=n_repeats, stratified=True,
        pooling=pooling, tune_metric="val_auroc", use_amp=False, freeze_backbone=0,
        num_classes=2, img_size=8, batch_size=4, dataloader_workers=0,
        lr=1e-3, weight_decay=1e-4, label_smoothing=0.0, drop_rate=0.0,
        randaugment_magnitude=0, randaugment_num_ops=1,
        mixup_alpha=0.0, cutmix_alpha=0.0, optimizer="AdamW",
    )
    cvic_mod._cvic_trial(config)
    assert rec.metrics is not None, "_cvic_trial never called tune.report"
    return cvic_mod, rec.metrics


def _expected_per_repeat_aurocs(cvic_mod, n_repeats):
    """Reconstruct each repeat's full-dev pooled OOF AUROC using cvic's OWN metric
    fn, so any AUROC convention (binary/OvR) matches exactly. Fold partition is
    irrelevant to the pooled value."""
    labels = _LABELS
    out = []
    for r in range(n_repeats):
        s = np.array([_score(i, r) for i in range(_N)], dtype=np.float32)
        p1 = 1.0 / (1.0 + np.exp(-s))                 # softmax([0, s]) class-1 prob
        probs = np.stack([1.0 - p1, p1], axis=1)
        out.append(cvic_mod._compute_metric(probs, labels, "auroc"))
    return np.array(out)


def test_pooling_reports_mean_and_sd_of_per_repeat_oof(monkeypatch, tmp_path):
    n_repeats = 4
    cvic_mod, m = _run_trial(monkeypatch, tmp_path, pooling=True, n_repeats=n_repeats)

    per_repeat = _expected_per_repeat_aurocs(cvic_mod, n_repeats)
    exp_mean = float(np.mean(per_repeat))
    exp_sd = float(np.std(per_repeat))

    # The reported development AUROC is the MEAN of per-repeat pooled OOF AUROCs...
    assert m["val_auroc"] == pytest.approx(exp_mean, abs=1e-6)
    assert m["val_auroc_sd"] == pytest.approx(exp_sd, abs=1e-6)
    # ...which the perturbation makes genuinely vary across repeats (real sd), and
    # is NOT the cross-repeat ensemble AUROC that the old --pooling reported.
    assert exp_sd > 1e-3, "test setup should induce across-repeat variation"
    p1_ens = np.mean(
        [1.0 / (1.0 + np.exp(-np.array([_score(i, r) for i in range(_N)])))
         for r in range(n_repeats)], axis=0)
    ensemble = cvic_mod._compute_metric(np.stack([1 - p1_ens, p1_ens], axis=1),
                                        _LABELS, "auroc")
    assert m["val_auroc"] != pytest.approx(ensemble, abs=1e-6), \
        "reported AUROC matches the old ensemble form — regression"


def test_default_reports_per_fold_mean_and_nan_sd(monkeypatch, tmp_path):
    cvic_mod, m = _run_trial(monkeypatch, tmp_path, pooling=False, n_repeats=4)

    # Default mode carries no across-repeat band.
    assert math.isnan(m["val_auroc_sd"])
    # And a valid averaged development AUROC.
    assert 0.0 <= m["val_auroc"] <= 1.0
    assert 0.0 <= m["val_acc"] <= 1.0
