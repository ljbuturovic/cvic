"""Tests for group-aware holdout splitting (_split_by_label with item_group).

Mirrors the grouped-path guarantees proven for the CV seam in test_cv_folds.py,
but for the tunic F/R holdout split. The invariant that matters: when items are
grouped (e.g. all images of one BHAM lesion share a group id), no group is ever
split across train and val, so there is no lesion-level leakage into the HPO
validation signal.
"""

import pytest

from cvic.tunic import _split_by_label


# 2 classes; class 0 = a0..a59 in 30 pairs, class 1 = b0..b39 in 20 pairs.
# Every group is class-pure (one label per group).
def _fixture():
    by_label = {0: [f"a{i:02d}" for i in range(60)],
                1: [f"b{i:02d}" for i in range(40)]}
    item_group = {}
    for i in range(60):
        item_group[f"a{i:02d}"] = f"ga{i // 2}"   # pairs share a group
    for i in range(40):
        item_group[f"b{i:02d}"] = f"gb{i // 2}"
    label_of = {it: lab for lab, items in by_label.items() for it in items}
    return by_label, item_group, label_of


def test_grouped_holdout_no_group_straddles_train_val():
    by_label, item_group, _ = _fixture()
    train, val, _ = _split_by_label(by_label, 0.8, 0.2, seed=1, item_group=item_group)
    train_groups = {item_group[it] for it in train}
    val_groups = {item_group[it] for it in val}
    assert train_groups.isdisjoint(val_groups), "a group straddled train/val"
    # and no item appears on both sides
    assert set(train).isdisjoint(val)


def test_grouped_holdout_preserves_both_classes_each_side():
    by_label, item_group, label_of = _fixture()
    train, val, _ = _split_by_label(by_label, 0.8, 0.2, seed=1, item_group=item_group)
    assert {label_of[it] for it in train} == {0, 1}
    assert {label_of[it] for it in val} == {0, 1}, "a class went missing from val"


def test_grouped_holdout_val_nonempty_and_approximates_fraction():
    by_label, item_group, _ = _fixture()
    train, val, dist = _split_by_label(by_label, 0.8, 0.2, seed=3, item_group=item_group)
    assert len(val) > 0 and len(train) > 0
    # dist reports realized (not target) counts for the grouped path
    for lab in (0, 1):
        assert dist[lab]["train"] == sum(1 for it in train if it in by_label[lab])
        assert dist[lab]["val"] == sum(1 for it in val if it in by_label[lab])
    # val fraction is in a sane neighborhood of 0.2 given atomic groups of size 2
    frac = len(val) / (len(train) + len(val))
    assert 0.1 <= frac <= 0.35


def test_grouped_holdout_reproducible_and_seed_sensitive():
    by_label, item_group, _ = _fixture()
    a = _split_by_label(by_label, 0.8, 0.2, seed=5, item_group=item_group)
    b = _split_by_label(by_label, 0.8, 0.2, seed=5, item_group=item_group)
    c = _split_by_label(by_label, 0.8, 0.2, seed=6, item_group=item_group)
    assert a[0] == b[0] and a[1] == b[1], "same seed must give identical grouped split"
    assert (a[0], a[1]) != (c[0], c[1]), "different seed must give a different split"


def test_grouped_holdout_rejects_class_straddling_group():
    """A group whose items span two classes must raise, not silently leak."""
    by_label = {0: ["x0", "x1"], 1: ["x2", "x3"]}
    # 'x1' and 'x2' share a group id but sit in different classes.
    item_group = {"x0": "g0", "x1": "gX", "x2": "gX", "x3": "g1"}
    with pytest.raises(ValueError, match="span multiple classes"):
        _split_by_label(by_label, 0.8, 0.2, seed=0, item_group=item_group)


def test_ungrouped_still_matches_when_item_group_none():
    """Passing item_group=None is exactly the historical per-item split."""
    by_label = {0: [f"a{i:02d}" for i in range(20)],
                1: [f"b{i:02d}" for i in range(20)]}
    grouped = _split_by_label(by_label, 0.8, 0.2, seed=42, item_group=None)
    plain = _split_by_label(by_label, 0.8, 0.2, seed=42)
    assert grouped == plain
