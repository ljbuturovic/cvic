"""Golden / characterization tests for tunic's train/val split paths.

These freeze the EXACT current output of the two split implementations that
production actually runs, so that an upcoming refactor (consolidating the
duplicated split logic into a single shared, group-aware seam) cannot silently
change how RSNA / Tiny ImageNet / BHAM data is partitioned. As long as these
stay green, the split is byte-identical to the pre-refactor behavior and no
result CSVs need to be regenerated.

The golden values were captured by running the shipped code on the fixtures
defined here (fixed seeds, fixed synthetic data). They are deterministic:
ImageFolder scans classes and filenames in sorted order, and both split
functions draw from a seeded random.Random. If a change here is intentional,
re-capture the values deliberately — do not "fix" a red test by pasting new
numbers without understanding why the split moved.

Covered paths:
  * _build_loaders           — imagefolder, no val/ dir  (BHAM F/R runs this)
  * _stratified_disjoint_split / _split_by_label — webdataset, no val split
"""

import numpy as np
import pytest
from PIL import Image
from torchvision.transforms import ToTensor


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_imagefolder(root, classes, n_per_class):
    """Zero-padded filenames so ImageFolder's string sort == numeric order.

    The split decision reads only base_dataset.samples (paths + labels), never
    pixels, so blank 4x4 images are sufficient and keep the fixture fast.
    """
    for cls in classes:
        d = root / "train" / cls
        d.mkdir(parents=True)
        for i in range(n_per_class):
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(d / f"img_{i:03d}.jpg")


# ---------------------------------------------------------------------------
# IMAGEFOLDER path — _build_loaders, no val/ dir
# ---------------------------------------------------------------------------

# (classes, n_per_class, seed, training_fraction, val_fraction) -> (train_idx, val_idx)
_IMAGEFOLDER_GOLDEN = {
    ("a", "b", "c", 20, 42, 0.5, 0.3): (
        [4, 5, 6, 9, 12, 13, 14, 15, 18, 19, 21, 23, 25, 28, 30, 31, 32, 34, 36,
         38, 40, 42, 44, 45, 49, 50, 52, 53, 54, 59],
        [1, 2, 10, 11, 16, 17, 20, 22, 24, 27, 29, 35, 41, 46, 47, 55, 56, 57],
    ),
    ("a", "b", "c", 20, 7, 0.8, 0.2): (
        [0, 2, 3, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25,
         26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 45, 46, 47,
         48, 50, 51, 52, 53, 54, 55, 56, 58, 59],
        [1, 4, 10, 12, 21, 22, 33, 37, 43, 44, 49, 57],
    ),
    ("a", "b", 10, 123, 0.7, 0.3): (
        [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17],
        [0, 1, 4, 15, 18, 19],
    ),
}


@pytest.mark.parametrize("key,expected", list(_IMAGEFOLDER_GOLDEN.items()))
def test_build_loaders_imagefolder_split_is_frozen(tmp_path, key, expected):
    from cvic.tunic import _build_loaders

    *classes, n_per_class, seed, tf, vf = key
    _make_imagefolder(tmp_path, classes, n_per_class)
    train_loader, val_loader, _ = _build_loaders(
        tmp_path, batch_size=4, workers=0, seed=seed,
        train_tf=ToTensor(), val_tf=ToTensor(),
        training_fraction=tf, val_fraction=vf,
    )
    train_idx = sorted(train_loader.dataset.indices)
    val_idx = sorted(val_loader.dataset.indices)
    exp_train, exp_val = expected
    assert train_idx == exp_train, "imagefolder train split moved from frozen baseline"
    assert val_idx == exp_val, "imagefolder val split moved from frozen baseline"
    # invariants that must hold regardless of the exact values
    assert set(train_idx).isdisjoint(val_idx)


# ---------------------------------------------------------------------------
# WEBDATASET path — _stratified_disjoint_split (returns key SETS)
# ---------------------------------------------------------------------------

def test_stratified_disjoint_split_membership_is_frozen():
    from cvic.tunic import _stratified_disjoint_split

    samples = [(f"s{i:04d}", i % 2) for i in range(100)]
    train_keys, val_keys, dist = _stratified_disjoint_split(samples, 0.8, 0.2, seed=42)

    exp_val = ['s0002', 's0010', 's0011', 's0012', 's0014', 's0015', 's0016',
               's0028', 's0030', 's0034', 's0037', 's0047', 's0049', 's0068',
               's0069', 's0071', 's0073', 's0079', 's0080', 's0081']
    assert sorted(val_keys) == exp_val, "webdataset val membership moved from baseline"
    # train is exactly the complement of val over the 100 samples, minus nothing
    assert len(train_keys) == 80
    assert train_keys.isdisjoint(val_keys)
    assert dist == {0: {'total': 50, 'train': 40, 'val': 10},
                    1: {'total': 50, 'train': 40, 'val': 10}}


def test_stratified_disjoint_split_membership_is_frozen_uneven():
    from cvic.tunic import _stratified_disjoint_split

    samples = [(f"s{i:04d}", i % 2) for i in range(90)]
    train_keys, val_keys, dist = _stratified_disjoint_split(samples, 0.7, 0.3, seed=123)

    exp_val = ['s0001', 's0004', 's0006', 's0010', 's0012', 's0020', 's0027',
               's0034', 's0039', 's0041', 's0042', 's0043', 's0048', 's0051',
               's0052', 's0055', 's0061', 's0063', 's0065', 's0067', 's0068',
               's0070', 's0072', 's0073', 's0077', 's0085', 's0086', 's0088']
    assert sorted(val_keys) == exp_val
    assert train_keys.isdisjoint(val_keys)
    assert dist == {0: {'total': 45, 'train': 31, 'val': 14},
                    1: {'total': 45, 'train': 31, 'val': 14}}


# ---------------------------------------------------------------------------
# WEBDATASET path — _split_by_label (ORDERED lists; pins RNG draw order exactly)
# ---------------------------------------------------------------------------

def test_split_by_label_order_is_frozen():
    """The ordered output pins the exact RNG shuffle sequence per class, which
    the set-based test above cannot see. A refactor that preserves membership
    but perturbs draw order would still be caught here."""
    from cvic.tunic import _split_by_label

    by_label = {0: [f"a{i:04d}" for i in range(60)],
                1: [f"b{i:04d}" for i in range(40)]}
    train, val, dist = _split_by_label(by_label, 0.8, 0.2, seed=0)

    exp_val = ['a0030', 'a0019', 'a0025', 'a0031', 'a0032', 'a0016', 'a0002',
               'a0026', 'a0056', 'a0048', 'a0024', 'a0054', 'b0032', 'b0020',
               'b0035', 'b0005', 'b0028', 'b0034', 'b0009', 'b0015']
    assert val == exp_val, "per-class RNG draw order moved from frozen baseline"
    assert train[:5] == ['a0012', 'a0033', 'a0029', 'a0036', 'a0044']
    assert len(train) == 80 and len(val) == 20
    assert dist == {0: {'total': 60, 'train': 48, 'val': 12},
                    1: {'total': 40, 'train': 32, 'val': 8}}
