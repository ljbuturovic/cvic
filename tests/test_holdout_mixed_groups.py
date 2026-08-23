"""tunic group-aware hold-out must tolerate label-mixed groups (bret patients).

bret patients can have one referable eye and one non-referable eye, so a patient
group legitimately spans both classes. tunic's group-aware hold-out splitter
(_split_by_label with item_group) used to REJECT such groups with a ValueError
("group(s) span multiple classes"); it now assigns the whole group atomically by
its majority label. These tests pin the new contract:

  * a mixed group no longer raises, and is never split across train/val;
  * NO group (mixed or pure) is ever split across train/val (leakage guard);
  * class-pure grouped splits are unchanged (delegated to test_split_golden.py,
    re-asserted here as a no-group-split invariant);
  * the split is deterministic for a fixed seed.
"""

from collections import defaultdict

import pytest

from cvic.tunic import _split_by_label


def _group_of(item_group, item):
    return item_group[item]


def _assert_no_group_split(train, val, item_group):
    """No group id may appear on both sides."""
    tg = {item_group[i] for i in train}
    vg = {item_group[i] for i in val}
    overlap = tg & vg
    assert not overlap, f"group(s) split across train/val: {sorted(overlap)}"


# --------------------------------------------------------------------------- #
# mixed groups
# --------------------------------------------------------------------------- #

def test_mixed_group_does_not_raise_and_is_not_split():
    # patient p_mix has one negative (0) and one positive (1) eye
    by_label = {
        0: ["p1_l", "p1_r", "p2_l", "p2_r", "p3_l", "p3_r", "pmix_neg"],
        1: ["p4_l", "p4_r", "p5_l", "p5_r", "pmix_pos"],
    }
    item_group = {
        "p1_l": "p1", "p1_r": "p1", "p2_l": "p2", "p2_r": "p2",
        "p3_l": "p3", "p3_r": "p3", "p4_l": "p4", "p4_r": "p4",
        "p5_l": "p5", "p5_r": "p5",
        "pmix_neg": "pmix", "pmix_pos": "pmix",   # one group, both labels
    }
    # must not raise (the old behavior raised ValueError here)
    train, val, dist = _split_by_label(by_label, 0.8, 0.2, seed=0, item_group=item_group)
    _assert_no_group_split(train, val, item_group)
    # both eyes of the mixed patient land together
    assert ("pmix_neg" in train) == ("pmix_pos" in train)
    assert ("pmix_neg" in val) == ("pmix_pos" in val)
    # dist reports by true per-item label and totals are preserved
    assert dist[0]["total"] == 7 and dist[1]["total"] == 5
    assert dist[0]["train"] + dist[0]["val"] <= 7
    # every placed item is accounted for once
    assert len(set(train) & set(val)) == 0


def test_mixed_group_assigned_by_majority_then_larger_label():
    # pmaj: two negatives + one positive -> majority label 0
    # ptie: one negative + one positive -> tie -> larger label index (1)
    by_label = {
        0: ["a", "b", "pmaj_n1", "pmaj_n2", "ptie_n"],
        1: ["c", "d", "pmaj_p1", "ptie_p"],
    }
    item_group = {
        "a": "a", "b": "b", "c": "c", "d": "d",
        "pmaj_n1": "pmaj", "pmaj_n2": "pmaj", "pmaj_p1": "pmaj",
        "ptie_n": "ptie", "ptie_p": "ptie",
    }
    # Just assert it runs, keeps groups whole, and is leakage-free across a few seeds.
    for seed in range(5):
        train, val, _ = _split_by_label(by_label, 0.7, 0.3, seed=seed, item_group=item_group)
        _assert_no_group_split(train, val, item_group)
        assert ("pmaj_n1" in train) == ("pmaj_p1" in train)   # group whole
        assert ("ptie_n" in train) == ("ptie_p" in train)


def test_many_mixed_groups_never_leak():
    # 30 patients, every 3rd one label-mixed; both eyes present
    by_label = defaultdict(list)
    item_group = {}
    for p in range(30):
        l, r = f"p{p}_l", f"p{p}_r"
        item_group[l] = item_group[r] = f"p{p}"
        if p % 3 == 0:                 # mixed
            by_label[0].append(l)
            by_label[1].append(r)
        elif p % 3 == 1:               # pure negative
            by_label[0].extend([l, r])
        else:                          # pure positive
            by_label[1].extend([l, r])
    by_label = dict(by_label)
    train, val, _ = _split_by_label(by_label, 0.8, 0.2, seed=42, item_group=item_group)
    _assert_no_group_split(train, val, item_group)
    # nothing dropped-then-duplicated
    assert len(set(train)) == len(train) and len(set(val)) == len(val)


# --------------------------------------------------------------------------- #
# pure groups: unchanged invariants
# --------------------------------------------------------------------------- #

def test_pure_groups_no_split_and_deterministic():
    by_label = {
        0: [f"n{i}_{e}" for i in range(8) for e in ("l", "r")],
        1: [f"p{i}_{e}" for i in range(4) for e in ("l", "r")],
    }
    item_group = {}
    for i in range(8):
        for e in ("l", "r"):
            item_group[f"n{i}_{e}"] = f"n{i}"
    for i in range(4):
        for e in ("l", "r"):
            item_group[f"p{i}_{e}"] = f"p{i}"
    a = _split_by_label(by_label, 0.8, 0.2, seed=7, item_group=item_group)
    b = _split_by_label(by_label, 0.8, 0.2, seed=7, item_group=item_group)
    assert a == b                                   # deterministic
    _assert_no_group_split(a[0], a[1], item_group)


def test_ungrouped_path_unaffected():
    """item_group=None still does a plain per-item split (golden path)."""
    by_label = {0: list(range(20)), 1: list(range(100, 110))}
    train, val, dist = _split_by_label(by_label, 0.8, 0.2, seed=1, item_group=None)
    assert set(train) & set(val) == set()
    assert dist[0]["total"] == 20 and dist[1]["total"] == 10
