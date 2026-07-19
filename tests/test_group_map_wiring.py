"""End-to-end wiring tests for --group-map in tunic.

Proves the group map flows CLI -> loader -> split and actually yields a
group-disjoint (lesion-disjoint) train/val partition, and that without it the
same data leaks groups across the split (the defect the feature fixes).
"""

import csv

import numpy as np
import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from cvic.common_cvic import _load_group_map
from cvic.tunic import _build_loaders, parse_args


def _make_grouped_imagefolder(root, n_lesions_per_class=30, imgs_per_lesion=2):
    """train/<class>/<lesion>_<k>.jpg, several images sharing one lesion id.

    Returns the {image_stem: lesion_id} mapping and writes a manifest CSV.
    """
    mapping = {}
    for cls in ("benign", "malignant"):
        d = root / "train" / cls
        d.mkdir(parents=True)
        for les in range(n_lesions_per_class):
            lesion_id = f"{cls[:3]}_les{les:03d}"
            for k in range(imgs_per_lesion):
                stem = f"{lesion_id}_{k}"
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(d / f"{stem}.jpg")
                mapping[stem] = lesion_id
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "lesion_id"])
        for stem, les in mapping.items():
            w.writerow([stem, les])
    return mapping, manifest


def _val_train_groups(loader_indices, base_samples, group_map):
    return {group_map[__stem(base_samples[i][0])] for i in loader_indices}


def __stem(path):
    from pathlib import Path
    return Path(path).stem


# ---------------------------------------------------------------------------
# _load_group_map parsing
# ---------------------------------------------------------------------------

def test_load_group_map_manifest_style(tmp_path):
    _, manifest = _make_grouped_imagefolder(tmp_path, n_lesions_per_class=3, imgs_per_lesion=2)
    m = _load_group_map(str(manifest))
    assert len(m) == 12                     # 2 classes * 3 lesions * 2 imgs
    assert len(set(m.values())) == 6        # 6 distinct lesions
    assert m["ben_les000_0"] == "ben_les000"


def test_load_group_map_plain_two_column(tmp_path):
    f = tmp_path / "g.csv"
    f.write_text("img_a.png,groupX\nimg_b.png,groupX\nimg_c.png,groupY\n")
    m = _load_group_map(str(f))
    # no recognizable header -> first two columns are data; stems drop extension
    assert m == {"img_a": "groupX", "img_b": "groupX", "img_c": "groupY"}


def test_load_group_map_none():
    assert _load_group_map(None) is None


# ---------------------------------------------------------------------------
# CLI arg reaches args
# ---------------------------------------------------------------------------

def test_cli_group_map(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv",
                        ["tunic", "--data", "/tmp", "--model", "resnet18",
                         "--group-map", "/tmp/manifest.csv"])
    args = parse_args()
    assert args.group_map == "/tmp/manifest.csv"


# ---------------------------------------------------------------------------
# End-to-end: the split is group-disjoint with the map, leaky without it
# ---------------------------------------------------------------------------

def test_build_loaders_group_map_yields_lesion_disjoint_split(tmp_path):
    mapping, manifest = _make_grouped_imagefolder(tmp_path)
    group_map = _load_group_map(str(manifest))

    train_loader, val_loader, _ = _build_loaders(
        tmp_path, batch_size=4, workers=0, seed=1,
        train_tf=ToTensor(), val_tf=ToTensor(),
        training_fraction=0.7, val_fraction=0.3,
        group_map=group_map,
    )
    base_samples = train_loader.dataset.dataset.samples
    train_groups = _val_train_groups(train_loader.dataset.indices, base_samples, mapping)
    val_groups = _val_train_groups(val_loader.dataset.indices, base_samples, mapping)
    assert train_groups.isdisjoint(val_groups), "a lesion straddled train/val despite --group-map"


def test_build_loaders_without_group_map_leaks_groups(tmp_path):
    """Contrast: the identical data with no map splits some lesion across
    train/val — the leakage the feature exists to prevent."""
    mapping, _ = _make_grouped_imagefolder(tmp_path)

    train_loader, val_loader, _ = _build_loaders(
        tmp_path, batch_size=4, workers=0, seed=1,
        train_tf=ToTensor(), val_tf=ToTensor(),
        training_fraction=0.7, val_fraction=0.3,
        group_map=None,
    )
    base_samples = train_loader.dataset.dataset.samples
    train_groups = _val_train_groups(train_loader.dataset.indices, base_samples, mapping)
    val_groups = _val_train_groups(val_loader.dataset.indices, base_samples, mapping)
    assert train_groups & val_groups, "expected ungrouped split to leak at least one lesion"
