"""Unit tests for the shared in-memory dataset path in common_cvic.

These cover load_wds_images + CachedImageDataset, the decode-once / augment-fresh
pipeline that both tunic and cvic use. The key regression guarded here is that
augmentation is re-sampled on every access (fresh per epoch) and NOT frozen at
load time — the bug that previously made cvic's webdataset training apply a single
static augmentation across all epochs and folds.
"""
import io
import json
import tarfile

import numpy as np
import torch
from PIL import Image


def _make_wds_dataset(root, classes, per_class=6, size=32):
    """Write a minimal single-shard WebDataset under <root>/wds/train/."""
    wds_dir = root / "wds"
    split_dir = wds_dir / "train"
    split_dir.mkdir(parents=True)

    shard = split_dir / "shard-000000.tar"
    n = 0
    with tarfile.open(shard, "w") as tar:
        for ci, cls in enumerate(classes):
            for i in range(per_class):
                key = f"{cls}_{i:03d}"
                arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                png_bytes = buf.getvalue()
                ti = tarfile.TarInfo(f"{key}.png")
                ti.size = len(png_bytes)
                tar.addfile(ti, io.BytesIO(png_bytes))
                cls_bytes = cls.encode()
                tc = tarfile.TarInfo(f"{key}.cls")
                tc.size = len(cls_bytes)
                tar.addfile(tc, io.BytesIO(cls_bytes))
                n += 1

    info = {
        "classes": list(classes),
        "splits": {"train": {"num_shards": 1, "num_samples": n}},
    }
    with open(wds_dir / "dataset_info.json", "w") as f:
        json.dump(info, f)
    return n


def test_load_wds_images_decodes_to_pil(tmp_path):
    from cvic.common_cvic import load_wds_images

    classes = ["cat", "dog", "bird"]
    n = _make_wds_dataset(tmp_path, classes, per_class=5)

    images, labels, num_classes = load_wds_images(str(tmp_path), split="train")

    assert num_classes == 3
    assert len(images) == n
    assert len(labels) == n
    # Decoded, not transformed: raw PIL RGB images.
    assert all(isinstance(im, Image.Image) for im in images)
    assert all(im.mode == "RGB" for im in images)
    # String class labels mapped to integer indices in range.
    assert set(labels) == {0, 1, 2}


def test_cached_dataset_augments_fresh_each_access(tmp_path):
    """The whole point of the fix: a random (train) transform must produce a
    DIFFERENT tensor on each access of the same index — augmentation is not frozen."""
    from cvic.common_cvic import load_wds_images, CachedImageDataset, build_transforms

    _make_wds_dataset(tmp_path, ["a", "b"], per_class=4, size=32)
    images, labels, _ = load_wds_images(str(tmp_path), split="train")

    train_tf = build_transforms(img_size=32, randaug_magnitude=9, randaug_num_ops=2, is_train=True)
    ds = CachedImageDataset(images, labels, train_tf)

    x1, y1 = ds[0]
    x2, y2 = ds[0]
    assert y1 == y2                      # same underlying sample / label
    assert x1.shape == x2.shape
    assert not torch.allclose(x1, x2)    # but augmentation re-sampled -> different pixels


def test_cached_dataset_val_transform_is_deterministic(tmp_path):
    """A clean (val) transform must be reproducible across accesses."""
    from cvic.common_cvic import load_wds_images, CachedImageDataset, build_transforms

    _make_wds_dataset(tmp_path, ["a", "b"], per_class=4, size=32)
    images, labels, _ = load_wds_images(str(tmp_path), split="train")

    val_tf = build_transforms(img_size=32, is_train=False)
    ds = CachedImageDataset(images, labels, val_tf)

    x1, _ = ds[0]
    x2, _ = ds[0]
    assert torch.allclose(x1, x2)        # no randomness in the eval transform


def test_cached_dataset_shares_images_between_views(tmp_path):
    """Train and val views can share one decoded image list (no duplication)."""
    from cvic.common_cvic import load_wds_images, CachedImageDataset, build_transforms

    _make_wds_dataset(tmp_path, ["a", "b"], per_class=4, size=32)
    images, labels, _ = load_wds_images(str(tmp_path), split="train")

    aug = CachedImageDataset(images, labels, build_transforms(32, 9, 2, is_train=True))
    clean = CachedImageDataset(images, labels, build_transforms(32, is_train=False))

    assert aug.images is clean.images
    assert len(aug) == len(clean) == len(images)
