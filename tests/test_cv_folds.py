"""Unit + golden tests for cvic's CV fold seam (_make_folds).

The fold construction used to live inline inside _cvic_trial with no unit
coverage — only the slow end-to-end smoke tests exercised it. It is now a single
extractable seam. These tests pin its behavior:

  * golden  — the ungrouped stratified split is frozen byte-for-byte, so the
              extraction is provably identical to the old inline StratifiedKFold
              and no cvic result CSVs need regenerating.
  * invariants — disjointness, out-of-fold coverage (every sample validated
              exactly once per repeat), per-fold class presence, reproducibility.
  * grouped — with groups given, no group id straddles train/val in any fold
              (the leakage protection for lesion-grouped data like BHAM).
"""

import numpy as np
import pytest

from cvic.cvic import _make_folds


# ---------------------------------------------------------------------------
# GOLDEN — ungrouped stratified split frozen to the historical inline behavior
# ---------------------------------------------------------------------------

# N=30, 3 balanced classes (label = i % 3), n_folds=5, shuffle=True.
# Values captured from StratifiedKFold(shuffle=True, random_state=seed) as used
# by the pre-extraction inline code.
_GOLDEN_VAL_FOLDS = {
    0: [[7, 12, 17, 19, 20, 24],
        [0, 1, 2, 5, 10, 21],
        [4, 6, 11, 14, 27, 28],
        [15, 18, 22, 23, 25, 29],
        [3, 8, 9, 13, 16, 26]],
    42: [[1, 3, 4, 8, 9, 26],
         [5, 13, 15, 20, 24, 28],
         [6, 10, 16, 17, 21, 29],
         [11, 12, 19, 23, 25, 27],
         [0, 2, 7, 14, 18, 22]],
}


@pytest.mark.parametrize("seed", [0, 42])
def test_make_folds_ungrouped_stratified_is_frozen(seed):
    labels = [i % 3 for i in range(30)]
    splits = _make_folds(30, labels, n_folds=5, stratified=True, seed=seed)
    val_folds = [sorted(int(i) for i in val_idx) for _, val_idx in splits]
    assert val_folds == _GOLDEN_VAL_FOLDS[seed], "cvic fold split moved from frozen baseline"


# ---------------------------------------------------------------------------
# INVARIANTS — hold regardless of the exact indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stratified", [True, False])
def test_make_folds_disjoint_and_full_coverage(stratified):
    """Each fold's train/val are disjoint, and across folds every sample is a
    validation sample exactly once (out-of-fold coverage)."""
    N, n_folds = 40, 5
    labels = [i % 2 for i in range(N)]
    splits = _make_folds(N, labels, n_folds=n_folds, stratified=stratified, seed=7)
    assert len(splits) == n_folds

    val_union = []
    for train_idx, val_idx in splits:
        train_set, val_set = set(int(i) for i in train_idx), set(int(i) for i in val_idx)
        assert train_set.isdisjoint(val_set)
        assert train_set | val_set == set(range(N))  # every sample is train or val
        val_union.extend(val_set)
    # every sample validated exactly once across the folds
    assert sorted(val_union) == list(range(N))


def test_make_folds_stratified_preserves_class_in_every_fold():
    """With stratification each class appears in every validation fold."""
    N, n_folds = 50, 5
    labels = [i % 2 for i in range(N)]  # two classes, 25 each
    splits = _make_folds(N, labels, n_folds=n_folds, stratified=True, seed=0)
    for _, val_idx in splits:
        val_labels = {labels[i] for i in val_idx}
        assert val_labels == {0, 1}, "a class went missing from a validation fold"


def test_make_folds_reproducible_and_seed_sensitive():
    labels = [i % 3 for i in range(30)]
    a = _make_folds(30, labels, 5, True, seed=123)
    b = _make_folds(30, labels, 5, True, seed=123)
    c = _make_folds(30, labels, 5, True, seed=456)
    same = all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    diff = any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c))
    assert same, "same seed must give identical folds"
    assert diff, "different seed must give different folds"


# ---------------------------------------------------------------------------
# GROUPED — the leakage protection: no group straddles train/val in any fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stratified", [True, False])
def test_make_folds_grouped_keeps_groups_intact(stratified):
    """Every sample sharing a group id lands in the same fold — so a grouped
    unit (e.g. all images of one BHAM lesion) is never split across train/val."""
    N, n_folds = 60, 5
    labels = [i % 2 for i in range(N)]
    # 30 groups of 2 samples each; both members share a group id.
    groups = [i // 2 for i in range(N)]
    splits = _make_folds(N, labels, n_folds=n_folds, stratified=stratified,
                         seed=1, groups=groups)
    assert len(splits) == n_folds
    for train_idx, val_idx in splits:
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups), "a group straddled train/val"


def test_make_folds_grouped_full_group_coverage():
    """Across folds every group is validated exactly once (OOF at group level)."""
    N, n_folds = 60, 5
    labels = [i % 2 for i in range(N)]
    groups = [i // 2 for i in range(N)]
    splits = _make_folds(N, labels, n_folds=n_folds, stratified=True,
                         seed=1, groups=groups)
    validated_groups = []
    for _, val_idx in splits:
        validated_groups.extend({groups[i] for i in val_idx})
    assert sorted(validated_groups) == sorted(set(groups))


def test_make_folds_grouped_reproducible():
    N = 60
    labels = [i % 2 for i in range(N)]
    groups = [i // 2 for i in range(N)]
    a = _make_folds(N, labels, 5, True, seed=9, groups=groups)
    b = _make_folds(N, labels, 5, True, seed=9, groups=groups)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
