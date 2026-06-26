#!/usr/bin/env -S python3 -B
"""tunic.py — Hyperparameter tuning for image classifiers using Ray Tune + timm."""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tunic")

warnings.filterwarnings("ignore", message=r".*with_length\(\).*", category=UserWarning, module="webdataset")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*", category=UserWarning)

# Set CUDA workspace config before any Ray workers spawn
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def _ck():
    """Import shared utilities (cached by sys.modules after first call)."""
    try:
        import common_cvic as _m
    except ImportError:
        import cvic.common_cvic as _m
    return _m


def _make_worker_init_fn(seed: int):
    """Return a worker_init_fn that seeds each DataLoader worker with a unique seed."""
    def worker_init_fn(worker_id: int):
        import random
        import numpy as np
        import torch
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


def _detect_format(data_root: str) -> str:
    """Return 'webdataset' or 'imagefolder' based on what's present at data_root."""
    if data_root.startswith("s3://"):
        return "webdataset"
    p = Path(data_root)
    if (p / "wds" / "dataset_info.json").exists():
        return "webdataset"
    return "imagefolder"


def _has_test_split(test_data: str | None) -> bool:
    """Return True if --test-data points at a valid test split.

    Two accepted layouts:
      1. WebDataset: <test_data>/wds/dataset_info.json with "test" in splits.
      2. ImageFolder: <test_data> itself is a dir of class subdirs.
    """
    if not test_data:
        return False
    if test_data.startswith("s3://"):
        return True
    p = Path(test_data)
    info = p / "wds" / "dataset_info.json"
    if info.exists():
        try:
            with open(info) as f:
                if "test" in json.load(f).get("splits", {}):
                    return True
        except Exception:
            pass
    if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
        return True
    return False


def _load_normalization_params(data_root: str) -> dict | None:
    """Load normalization params from dataset_info.json if available."""
    if data_root.startswith("s3://"):
        return None  # S3 support would require aws CLI
    p = Path(data_root)
    info_path = p / "wds" / "dataset_info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            meta = json.load(f)
        return meta.get("normalization")
    except Exception:
        return None


def _apply_dataset_normalization(norm_params: dict):
    """Return a normalization transform based on dataset params."""
    import torchvision.transforms as T
    if not norm_params or norm_params.get("type") != "pool_zscore_clipped":
        return None
    pool_mean = norm_params.get("pool_mean")
    pool_std = norm_params.get("pool_std")
    clip_sigma = norm_params.get("clip_sigma", 3.0)
    if pool_mean is None or pool_std is None:
        return None
    # Normalize encoded uint8 [0, 255] to [-clip_sigma, clip_sigma] range
    # Then divide by (2*clip_sigma) and add 0.5 to get [0, 1]
    # Then apply mean/std for the model
    return T.Normalize(mean=[0.5], std=[1.0 / (2.0 * clip_sigma)])


def _read_train_sample_labels(data_root: str, meta: dict, classes: list[str]) -> list[tuple[str, int]]:
    """Read all (sample_key, label) pairs from train shards (local or S3)."""
    import io
    import tarfile
    from collections import defaultdict

    is_s3 = data_root.startswith("s3://")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    n_shards = meta["splits"]["train"]["num_shards"]
    samples: list[tuple[str, int]] = []

    for i in range(n_shards):
        if is_s3:
            import subprocess
            shard_url = f"{data_root.rstrip('/')}/wds/train/shard-{i:06d}.tar"
            r = subprocess.run(["aws", "s3", "cp", shard_url, "-"],
                              capture_output=True, check=True)
            tf = tarfile.open(fileobj=io.BytesIO(r.stdout))
        else:
            shard_path = Path(data_root) / "wds" / "train" / f"shard-{i:06d}.tar"
            tf = tarfile.open(shard_path)
        try:
            for m in tf.getmembers():
                if not m.name.endswith(".cls"):
                    continue
                key = m.name[:-4]
                raw = tf.extractfile(m).read().decode().strip()
                try:
                    label = int(raw)
                    if not (0 <= label < n_classes):
                        label = class_to_idx[raw]
                except ValueError:
                    label = class_to_idx[raw]
                samples.append((key, label))
        finally:
            tf.close()
    return samples


def _split_by_label(by_label, training_fraction: float, val_fraction: float,
                    seed: int) -> tuple[list, list, dict]:
    """Per-class stratified disjoint split. Items may be keys or indices."""
    import random
    rng = random.Random(seed)
    train_items: list = []
    val_items: list = []
    dist: dict = {}
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        n_total = len(items)
        n_tr = max(1, int(round(n_total * training_fraction)))
        n_va = max(1, int(round(n_total * val_fraction)))
        if n_tr + n_va > n_total:
            n_va = max(1, n_total - n_tr)
        train_items.extend(items[:n_tr])
        val_items.extend(items[n_tr:n_tr + n_va])
        dist[label] = {"total": n_total, "train": n_tr, "val": n_va}
    return train_items, val_items, dist


def _stratified_disjoint_split(samples: list[tuple[str, int]],
                                training_fraction: float, val_fraction: float,
                                seed: int) -> tuple[set[str], set[str], dict]:
    """Stratified disjoint split of (key, label) pairs into train/val key sets."""
    from collections import defaultdict
    by_label: dict[int, list[str]] = defaultdict(list)
    for key, label in samples:
        by_label[label].append(key)
    train_keys, val_keys, dist = _split_by_label(by_label, training_fraction, val_fraction, seed)
    return set(train_keys), set(val_keys), dist


def _build_wds_loaders(data_root: str, batch_size: int, workers: int, seed: int,
                        train_tf, val_tf, collate_fn=None,
                        training_fraction: float = 1.0, val_fraction: float | None = None,
                        require_val: bool = True):
    """Build train/val DataLoaders from WebDataset TAR shards (local or s3://).

    Images are decoded into memory ONCE and the transform is applied lazily per
    sample (CachedImageDataset), so augmentation is re-sampled fresh every epoch.
    This replaces the previous per-epoch re-streaming pipeline, which re-read and
    re-decoded the shards on every epoch and left the GPU input-starved.
    """
    import random
    import torch
    from torch.utils.data import DataLoader
    ck = _ck()

    is_s3 = data_root.startswith("s3://")
    if is_s3:
        import subprocess
        info_url = data_root.rstrip("/") + "/wds/dataset_info.json"
        r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                           capture_output=True, text=True, check=True)
        meta = json.loads(r.stdout)
    else:
        with open(Path(data_root) / "wds" / "dataset_info.json") as f:
            meta = json.load(f)

    classes = meta["classes"]
    num_classes = len(classes)

    def _frac_subset(images, labels, fraction):
        """Seeded random subset of (images, labels) keeping `fraction` of samples."""
        if fraction >= 1.0:
            return images, labels
        n = len(images)
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        keep = idx[:max(batch_size, int(n * fraction))]
        return [images[i] for i in keep], [labels[i] for i in keep]

    def _make_loader(images, labels, tf, is_train):
        ds = ck.CachedImageDataset(images, labels, tf)
        gen = torch.Generator().manual_seed(seed)
        # pin_memory stays False: Ray Tune forks trial processes and the pin-memory
        # thread loses its IPC connection (see project notes).
        return DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=workers, pin_memory=False, drop_last=is_train,
            worker_init_fn=_make_worker_init_fn(seed) if workers > 0 else None,
            collate_fn=collate_fn if is_train else None,
            generator=gen if is_train else None,
        )

    # Decode the train split once (shared between the augmented and clean views).
    train_imgs, train_labels, _ = ck.load_wds_images(data_root, "train")

    if "val" in meta.get("splits", {}):
        val_imgs, val_labels, _ = ck.load_wds_images(data_root, "val")
        train_imgs, train_labels = _frac_subset(train_imgs, train_labels, training_fraction)
        vf = val_fraction if val_fraction is not None else 1.0
        val_imgs, val_labels = _frac_subset(val_imgs, val_labels, vf)
        train_loader = _make_loader(train_imgs, train_labels, train_tf, is_train=True)
        val_loader = _make_loader(val_imgs, val_labels, val_tf, is_train=False)
        return train_loader, val_loader, num_classes

    # No val split.
    if val_fraction is None:
        if not require_val:
            train_imgs, train_labels = _frac_subset(train_imgs, train_labels, training_fraction)
            train_loader = _make_loader(train_imgs, train_labels, train_tf, is_train=True)
            return train_loader, None, classes
        logger.error(
            "No val split in WebDataset and --val-fraction is not set. "
            "Specify --val-fraction to reserve a fraction of training data for validation "
            "(e.g. --val-fraction 0.2). "
            "--training-fraction + --val-fraction must be ≤ 1.0."
        )
        sys.exit(1)
    if training_fraction + val_fraction > 1.0:
        logger.error(
            f"--training-fraction ({training_fraction:.3f}) + --val-fraction ({val_fraction:.3f}) "
            f"= {training_fraction + val_fraction:.3f} > 1.0. "
            "When no val split exists, both fractions are drawn from the training set and must sum to ≤ 1.0."
        )
        sys.exit(1)

    # Stratified disjoint train/val carve from the in-memory train pool (by index).
    from collections import defaultdict
    logger.warning(f"[SPLIT] No val split — building stratified DISJOINT train/val split (seed={seed})")
    class_to_idxs = defaultdict(list)
    for i, lbl in enumerate(train_labels):
        class_to_idxs[lbl].append(i)

    train_idx, val_idx, dist = [], [], {}
    rng = random.Random(seed)
    for lbl in sorted(class_to_idxs):
        idxs = list(class_to_idxs[lbl])
        rng.shuffle(idxs)
        n_tr = max(1, round(len(idxs) * training_fraction))
        n_va = max(1, round(len(idxs) * val_fraction))
        tr, va = idxs[:n_tr], idxs[n_tr:n_tr + n_va]
        train_idx.extend(tr)
        val_idx.extend(va)
        dist[lbl] = {"train": len(tr), "val": len(va), "total": len(idxs)}

    overlap = set(train_idx) & set(val_idx)
    if overlap:
        logger.error(f"[SPLIT] BUG: train/val overlap detected — {len(overlap)} samples in both sets")
        sys.exit(1)
    logger.warning(
        f"[SPLIT] ✓ OVERLAP CHECK PASSED: train ({len(train_idx):,}) ∩ val ({len(val_idx):,}) "
        f"= 0 samples (verified disjoint)"
    )
    for label in sorted(dist.keys()):
        d = dist[label]
        cls_name = classes[label] if label < len(classes) else str(label)
        logger.warning(
            f"[SPLIT]   Class '{cls_name}': {d['train']:,} train + {d['val']:,} val "
            f"(of {d['total']:,} total)"
        )

    tr_imgs = [train_imgs[i] for i in train_idx]
    tr_lbls = [train_labels[i] for i in train_idx]
    va_imgs = [train_imgs[i] for i in val_idx]
    va_lbls = [train_labels[i] for i in val_idx]
    train_loader = _make_loader(tr_imgs, tr_lbls, train_tf, is_train=True)
    val_loader = _make_loader(va_imgs, va_lbls, val_tf, is_train=False)
    return train_loader, val_loader, num_classes


def _subsample(dataset, fraction: float, seed: int):
    """Return a Subset of dataset using a fixed fraction of its indices."""
    import random
    from torch.utils.data import Subset
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = list(dataset.indices if isinstance(dataset, Subset) else range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(base, indices[:max(1, int(len(indices) * fraction))])


def _build_loaders(data_path: Path, batch_size: int, workers: int, seed: int,
                   train_tf, val_tf, collate_fn=None,
                   training_fraction: float = 1.0, val_fraction: float | None = None,
                   require_val: bool = True):
    """Build train/val DataLoaders, creating a disjoint stratified split if val/ is absent."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if val_dir.exists():
        eff_val_frac = val_fraction if val_fraction is not None else 1.0
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)

        if training_fraction < 1.0:
            n_before = len(train_dataset)
            train_dataset = _subsample(train_dataset, training_fraction, seed)
            logger.info(f"Using {len(train_dataset)}/{n_before} training samples (training_fraction={training_fraction})")

        if eff_val_frac < 1.0:
            n_before = len(val_dataset)
            val_dataset = _subsample(val_dataset, eff_val_frac, seed)
            logger.info(f"Using {len(val_dataset)}/{n_before} val samples (val_fraction={eff_val_frac})")

    else:
        if val_fraction is None:
            if not require_val:
                train_base = datasets.ImageFolder(str(train_dir), transform=train_tf)
                if training_fraction < 1.0:
                    train_base = _subsample(train_base, training_fraction, seed)
                gen = torch.Generator().manual_seed(seed)
                loader = DataLoader(train_base, batch_size=batch_size, shuffle=True,
                                    num_workers=workers, pin_memory=True,
                                    worker_init_fn=_make_worker_init_fn(seed),
                                    collate_fn=collate_fn, generator=gen)
                return loader, None, train_base.dataset.classes if hasattr(train_base, 'dataset') else train_base.classes
            logger.error(
                "No val/ directory found and --val-fraction is not set. "
                "Specify --val-fraction to reserve a fraction of training images for validation "
                "(e.g. --val-fraction 0.2). "
                "--training-fraction + --val-fraction must be ≤ 1.0."
            )
            sys.exit(1)
        if training_fraction + val_fraction > 1.0:
            logger.error(
                f"--training-fraction ({training_fraction:.3f}) + --val-fraction ({val_fraction:.3f}) "
                f"= {training_fraction + val_fraction:.3f} > 1.0. "
                "When no val/ directory exists, both fractions are drawn from the same training pool "
                "and must sum to ≤ 1.0."
            )
            sys.exit(1)
        eff_val_frac = val_fraction

        from collections import defaultdict
        import random as _random
        logger.warning("No val/ directory — splitting train/ using training_fraction and val_fraction (disjoint)")
        base_dataset = datasets.ImageFolder(str(train_dir))

        class_to_idxs = defaultdict(list)
        for i, (_, lbl) in enumerate(base_dataset.samples):
            class_to_idxs[lbl].append(i)

        train_indices, val_indices = [], []
        rng = _random.Random(seed)
        for lbl in sorted(class_to_idxs):
            idxs = list(class_to_idxs[lbl])
            rng.shuffle(idxs)
            n_tr = max(1, round(len(idxs) * training_fraction))
            n_va = max(1, round(len(idxs) * eff_val_frac))
            train_indices.extend(idxs[:n_tr])
            val_indices.extend(idxs[n_tr:n_tr + n_va])

        train_base = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_base = datasets.ImageFolder(str(train_dir), transform=val_tf)
        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(val_base, val_indices)
        logger.info(f"Using {len(train_dataset)} training, {len(val_dataset)} val samples "
                    f"(disjoint from {len(base_dataset.samples)} total)")

    num_classes = len(train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
    pw = workers > 0
    worker_fn = _make_worker_init_fn(seed) if workers > 0 else None
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              persistent_workers=pw, worker_init_fn=worker_fn, collate_fn=collate_fn, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True, persistent_workers=pw,
                            worker_init_fn=worker_fn)
    return train_loader, val_loader, num_classes


def _get_class_names(data_root: str, fmt: str, num_classes: int) -> list[str]:
    """Return class name list for a dataset, falling back to generic names."""
    try:
        if fmt == "webdataset":
            with open(Path(data_root) / "wds" / "dataset_info.json") as f:
                names = json.load(f).get("classes")
            if names:
                return names
        else:
            from torchvision import datasets as tvd
            return tvd.ImageFolder(str(Path(data_root) / "train")).classes
    except Exception:
        pass
    return [f"class_{i}" for i in range(num_classes)]


def _save_probs_csv(model, loader, class_names: list[str], csv_path: str, device, use_amp: bool = False):
    """Run inference on loader and write sampleID,truth,prob_<class>,... CSV."""
    import csv
    import torch
    ck = _ck()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, dtype=ck.get_amp_dtype(), enabled=use_amp):
                outputs = model(images)
            all_probs.append(torch.softmax(outputs.float(), dim=1).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sampleID", "truth"] + [f"prob_{c}" for c in class_names])
        for i, (lbl, p) in enumerate(zip(labels, probs)):
            writer.writerow([i, class_names[int(lbl)]] + [f"{x:.6f}" for x in p])


def evaluate(model, loader, criterion, device, use_amp=False):
    import torch
    import numpy as np
    ck = _ck()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=ck.get_amp_dtype(), enabled=use_amp):
                outputs = model(images)
                loss_val = criterion(outputs, labels)
            total_loss += loss_val.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            all_probs.append(torch.softmax(outputs.float(), dim=1).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, ck._compute_auroc(probs, labels_np)


def _evaluate_distributed(model, loader, criterion, device, world_size: int):
    """Like evaluate(), but reduces loss/accuracy across Ray Train workers."""
    import torch
    import numpy as np
    ck = _ck()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            all_probs.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())

    if world_size > 1:
        import torch.distributed as dist
        stats = torch.tensor([correct, total, total_loss], dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        accuracy = stats[0].item() / stats[1].item()
        avg_loss = stats[2].item() / stats[1].item()
    else:
        accuracy = correct / total
        avg_loss = total_loss / total

    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return avg_loss, accuracy, ck._compute_auroc(probs, labels_np)


# ---------------------------------------------------------------------------
# Ray Train worker function
# ---------------------------------------------------------------------------

def train_func_distributed(config: dict):
    """Ray Train worker function. Receives all hyperparams + fixed config as a dict."""
    import torch.nn as nn
    import ray
    import ray.train
    import ray.train.torch
    ck = _ck()
    set_seed = ck.set_seed; build_transforms = ck.build_transforms
    MixupCutmixCollator = ck.MixupCutmixCollator; create_model = ck.create_model
    freeze_backbone = ck.freeze_backbone; unfreeze_all = ck.unfreeze_all
    get_optimizer = ck.get_optimizer; build_scheduler = ck.build_scheduler
    train_one_epoch = ck.train_one_epoch

    data_path = Path(config["data"])
    model_name = config["model"]
    pretrained = config["pretrained"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    freeze_backbone_epochs = config["freeze_backbone"]
    base_seed = config["seed"]
    dataloader_workers = config["dataloader_workers"]
    training_fraction = config["training_fraction"]
    val_fraction = config["val_fraction"]
    num_classes = config["num_classes"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    label_smoothing = config["label_smoothing"]
    drop_rate = config["drop_rate"]
    randaug_magnitude = config["randaugment_magnitude"]
    randaug_num_ops = config["randaugment_num_ops"]
    mixup_alpha = config["mixup_alpha"]
    cutmix_alpha = config["cutmix_alpha"]
    optimizer_name = config["optimizer"]
    use_amp = config.get("use_amp", False)

    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    if rank == 0:
        _counter = ray.get_actor("trial_counter")
        _trial_num = ray.get(_counter.next.remote())
    else:
        _trial_num = 0
    if world_size > 1:
        import torch.distributed as dist
        trial_num_tensor = torch.tensor([_trial_num], device="cpu")
        dist.broadcast(trial_num_tensor, src=0)
        _trial_num = int(trial_num_tensor.item())
    trial_id = f"{_trial_num}/{config['n_trials']}" if _trial_num else ""
    set_seed(base_seed + rank)
    device = ray.train.torch.get_device()

    shuffle_seed = config.get("shuffle_seed")
    split_seed = (shuffle_seed + _trial_num) if shuffle_seed is not None else base_seed

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(img_size, randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf = build_transforms(img_size, is_train=False)
    train_loader, val_loader, _ = _build_loaders(
        data_path, batch_size, dataloader_workers, split_seed,
        train_tf, val_tf, collate_fn,
        training_fraction=training_fraction, val_fraction=val_fraction,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    try:
        model = create_model(model_name, num_classes, pretrained, drop_rate)
        model = ray.train.torch.prepare_model(model)

        if freeze_backbone_epochs > 0:
            freeze_backbone(model)

        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = build_scheduler(optimizer, epochs, len(train_loader))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        val_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
                unfreeze_all(model)
                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                scheduler = build_scheduler(
                    optimizer, epochs, len(train_loader),
                    start_step=epoch * len(train_loader),
                )

            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                use_soft_labels=use_mixup_cutmix, trial_id=trial_id, epoch=epoch, epochs=epochs,
                use_amp=use_amp,
            )
            val_loss, val_acc, val_auroc = _evaluate_distributed(
                model, val_loader, val_criterion, device, world_size,
            )

            if rank == 0:
                logger.info(
                    f"epoch {epoch+1}/{epochs} | loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc:.4f}"
                )

            ray.train.report({"val_acc": val_acc, "val_auroc": val_auroc, "train_loss": train_loss})

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ray.train.report({"val_acc": 0.0, "val_auroc": float("nan"), "train_loss": float("inf")})


# ---------------------------------------------------------------------------
# Ray Tune plain-function trainable (single GPU/CPU per trial, no DDP)
# ---------------------------------------------------------------------------

def _tune_trial(config: dict):
    """Plain Ray Tune trainable. Each trial runs on one GPU (or CPU)."""
    import torch
    import torch.nn as nn
    import ray
    from ray import tune
    ck = _ck()
    get_device = ck.get_device; set_seed = ck.set_seed
    build_transforms = ck.build_transforms; MixupCutmixCollator = ck.MixupCutmixCollator
    create_model = ck.create_model; freeze_backbone = ck.freeze_backbone
    unfreeze_all = ck.unfreeze_all; get_optimizer = ck.get_optimizer
    build_scheduler = ck.build_scheduler; train_one_epoch = ck.train_one_epoch

    data_root   = config["data"]
    data_format = config.get("data_format", "imagefolder")
    data_path   = Path(data_root) if data_format == "imagefolder" else None
    device = get_device(config["device"])
    _counter = ray.get_actor("trial_counter")
    _trial_num = ray.get(_counter.next.remote())
    trial_id = f"{_trial_num}/{config['n_trials']}"
    set_seed(config["seed"])

    shuffle_seed = config.get("shuffle_seed")
    split_seed = (shuffle_seed + _trial_num) if shuffle_seed is not None else config["seed"]

    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    label_smoothing = config["label_smoothing"]
    drop_rate = config["drop_rate"]
    randaug_magnitude = config["randaugment_magnitude"]
    randaug_num_ops = config["randaugment_num_ops"]
    mixup_alpha = config["mixup_alpha"]
    cutmix_alpha = config["cutmix_alpha"]
    optimizer_name = config["optimizer"]
    num_classes = config["num_classes"]

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(config["img_size"], randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf = build_transforms(config["img_size"], is_train=False)

    # Apply custom normalization if available in dataset_info.json
    norm_params = config.get("normalization")
    if norm_params:
        import torchvision.transforms as T
        norm_tf = _apply_dataset_normalization(norm_params)
        if norm_tf:
            train_tf = T.Compose([train_tf, norm_tf])
            val_tf = T.Compose([val_tf, norm_tf])
    bs, workers = config["batch_size"], config["dataloader_workers"]
    tf = config["training_fraction"]
    vf = config["val_fraction"]
    if data_format == "webdataset":
        train_loader, val_loader, _ = _build_wds_loaders(
            data_root, bs, workers, split_seed, train_tf, val_tf, collate_fn, tf, vf)
    else:
        train_loader, val_loader, _ = _build_loaders(
            data_path, bs, workers, split_seed, train_tf, val_tf, collate_fn, tf, vf)

    try:
        model = create_model(config["model"], num_classes, config["pretrained"], drop_rate)
        model = model.to(device)

        if config["freeze_backbone"] > 0:
            freeze_backbone(model)

        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = build_scheduler(optimizer, epochs, len(train_loader))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        val_criterion = nn.CrossEntropyLoss()

        use_amp = config.get("use_amp", False)
        for epoch in range(epochs):
            if config["freeze_backbone"] > 0 and epoch == config["freeze_backbone"]:
                unfreeze_all(model)
                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                scheduler = build_scheduler(
                    optimizer, epochs, len(train_loader),
                    start_step=epoch * len(train_loader),
                )

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                use_soft_labels=use_mixup_cutmix, trial_id=trial_id, epoch=epoch, epochs=epochs,
                use_amp=use_amp,
            )
            _, val_acc, val_auroc = evaluate(model, val_loader, val_criterion, device, use_amp=use_amp)

            tune.report({"val_acc": val_acc, "val_auroc": val_auroc, "train_loss": train_loss})

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tune.report({"val_acc": 0.0, "val_auroc": float("nan"), "train_loss": float("inf")})


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------

def _build_combined_loader(data_root, fmt, batch_size, workers, seed, train_tf, collate_fn):
    """Build a DataLoader combining train+val splits. Exits if val is missing."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    if fmt == "webdataset":
        try:
            import webdataset as wds
        except ImportError:
            logger.error("webdataset not installed")
            sys.exit(1)
        is_s3 = data_root.startswith("s3://")
        if is_s3:
            import subprocess
            info_url = data_root.rstrip("/") + "/wds/dataset_info.json"
            r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                               capture_output=True, text=True, check=True)
            meta = json.loads(r.stdout)
        else:
            with open(Path(data_root) / "wds" / "dataset_info.json") as f:
                meta = json.load(f)
        if "val" not in meta.get("splits", {}):
            logger.error("--combine specified but no val split found in dataset_info.json")
            sys.exit(1)
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)

        def shard_urls(split):
            n = meta["splits"][split]["num_shards"]
            if is_s3:
                base = data_root.rstrip("/")
                return [f"pipe:aws s3 cp {base}/wds/{split}/shard-{i:06d}.tar -" for i in range(n)]
            d = Path(data_root) / "wds" / split
            return [str(d / f"shard-{i:06d}.tar") for i in range(n)]

        urls = shard_urls("train") + shard_urls("val")
        n_train = meta["splits"]["train"]["num_samples"]
        n_val = meta["splits"]["val"]["num_samples"]
        n_total = n_train + n_val if n_train > 0 and n_val > 0 else len(urls) * 5000

        _img_decoder = wds.autodecode.imagehandler("pil")
        def _decoder(key, data):
            if key.endswith(".cls"):
                return data
            return _img_decoder(key, data)
        def decode_cls(b):
            s = b.decode().strip()
            try:
                idx = int(s)
                if 0 <= idx < num_classes:
                    return idx
            except ValueError:
                pass
            return class_to_idx[s]
        def apply_tf(img):
            return train_tf(img.convert("RGB"))

        effective_workers = min(workers, len(urls)) if workers > 0 else 1
        slice_per_worker = max(batch_size, n_total // effective_workers)
        dataset = (
            wds.WebDataset(urls, shardshuffle=500, seed=seed,
                           nodesplitter=wds.split_by_node, empty_check=False)
            .shuffle(1000, seed=seed)
            .decode(_decoder)
            .to_tuple("png", "cls")
            .map_tuple(apply_tf, decode_cls)
            .slice(slice_per_worker)
            .with_length(n_total)
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                          pin_memory=False, worker_init_fn=_make_worker_init_fn(seed),
                          collate_fn=collate_fn), num_classes
    else:
        train_dir = Path(data_root) / "train"
        val_dir = Path(data_root) / "val"
        if not val_dir.exists():
            logger.error("--combine specified but no val/ directory found")
            sys.exit(1)
        from torch.utils.data import ConcatDataset
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_dir), transform=train_tf)
        num_classes = len(train_ds.classes)
        combined = ConcatDataset([train_ds, val_ds])
        worker_fn = _make_worker_init_fn(seed) if workers > 0 else None
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(combined, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          worker_init_fn=worker_fn, collate_fn=collate_fn, generator=gen), num_classes


def _build_test_loader(test_root, fmt, batch_size, workers, seed, val_tf):
    """Build a test DataLoader (WDS or imagefolder). Returns None if test_root is None.

    Two accepted layouts:
      1. WebDataset: <test_root>/wds/dataset_info.json with "test" in splits.
      2. ImageFolder: <test_root> itself is a dir of class subdirs.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets
    if test_root is None:
        return None
    if fmt == "webdataset":
        try:
            import webdataset as wds
        except ImportError:
            logger.error("webdataset not installed")
            return None
        is_s3 = test_root.startswith("s3://")
        if is_s3:
            import subprocess
            info_url = test_root.rstrip("/") + "/wds/dataset_info.json"
            r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                               capture_output=True, text=True, check=True)
            meta = json.loads(r.stdout)
        else:
            with open(Path(test_root) / "wds" / "dataset_info.json") as f:
                meta = json.load(f)
        if "test" not in meta.get("splits", {}):
            return None
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n_shards = meta["splits"]["test"]["num_shards"]
        n_samples = meta["splits"]["test"]["num_samples"]
        if is_s3:
            base = test_root.rstrip("/")
            urls = [f"pipe:aws s3 cp {base}/wds/test/shard-{i:06d}.tar -" for i in range(n_shards)]
        else:
            d = Path(test_root) / "wds" / "test"
            urls = [str(d / f"shard-{i:06d}.tar") for i in range(n_shards)]
        _img_decoder = wds.autodecode.imagehandler("pil")
        def _decoder(key, data):
            if key.endswith(".cls"):
                return data
            return _img_decoder(key, data)
        def decode_cls(b):
            s = b.decode().strip()
            try:
                idx = int(s)
                if 0 <= idx < len(classes):
                    return idx
            except ValueError:
                pass
            return class_to_idx[s]
        def apply_tf(img):
            return val_tf(img.convert("RGB"))
        effective_workers = min(workers, len(urls)) if workers > 0 else 1
        if n_samples > 0:
            slice_per_worker = max(batch_size, n_samples // effective_workers)
            length = n_samples
        else:
            slice_per_worker = 10 ** 9  # no limit
            length = n_shards * 5000    # rough estimate
        dataset = (
            wds.WebDataset(urls, shardshuffle=False, seed=seed,
                           nodesplitter=wds.split_by_node, empty_check=False)
            .decode(_decoder)
            .to_tuple("png", "cls")
            .map_tuple(apply_tf, decode_cls)
            .slice(slice_per_worker)
            .with_length(length)
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=False,
                          worker_init_fn=_make_worker_init_fn(seed))
    else:
        test_dir = Path(test_root)
        if not (test_dir.is_dir() and any(p.is_dir() for p in test_dir.iterdir())):
            return None
        test_dataset = datasets.ImageFolder(str(test_dir), transform=val_tf)
        worker_fn = _make_worker_init_fn(seed) if workers > 0 else None
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True, worker_init_fn=worker_fn)


def run_final(args):
    import torch
    import torch.nn as nn
    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm:
            @staticmethod
            def write(s): print(s)

    # Handle --cpu and --repro flags
    if hasattr(args, 'cpu') and args.cpu:
        args.device = "cpu"
        logger.info("Forcing CPU-only mode")

    if hasattr(args, 'repro') and args.repro:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Reproducibility mode enabled (may reduce GPU utilization)")
    elif hasattr(args, 'repro'):  # repro flag exists but is False
        torch.backends.cudnn.benchmark = True
        logger.info("Performance mode enabled (results may vary slightly across runs)")

    ck = _ck()
    validate_dataset_path = ck.validate_dataset_path
    get_device = ck.get_device
    set_seed = ck.set_seed
    get_amp_dtype = ck.get_amp_dtype
    MixupCutmixCollator = ck.MixupCutmixCollator
    build_transforms = ck.build_transforms
    create_model = ck.create_model
    freeze_backbone = ck.freeze_backbone
    unfreeze_all = ck.unfreeze_all
    get_optimizer = ck.get_optimizer
    build_scheduler = ck.build_scheduler
    train_one_epoch = ck.train_one_epoch
    t0 = time.time()
    final_json = Path(args.final)
    try:
        with open(final_json) as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"--final path does not exist: {final_json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in {final_json}: {e}")
        sys.exit(1)

    if "best_params" not in results:
        logger.error(f"No 'best_params' key in {final_json}")
        sys.exit(1)

    params = results["best_params"]
    model_name = args.model or results.get("model")
    if not model_name:
        logger.error("Model not found in results JSON and --model not specified. Pass --model explicitly.")
        sys.exit(1)
    num_classes = results.get("num_classes")
    data_root = args.data if args.data else results.get("dataset", ".")
    data_path = Path(data_root)
    epochs = args.epochs or results.get("epochs", 30)
    batch_size = params.get("batch_size", args.batch_size)

    validate_dataset_path(data_path)

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    use_mixup_cutmix = params.get("mixup_alpha", 0) > 0 or params.get("cutmix_alpha", 0) > 0
    collate_fn = MixupCutmixCollator(params.get("mixup_alpha", 0), params.get("cutmix_alpha", 0), num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(args.img_size, params.get("randaugment_magnitude", 0),
                                params.get("randaugment_num_ops", 2), is_train=True)
    val_tf = build_transforms(args.img_size, is_train=False)

    fmt = _detect_format(data_root)
    # Final training uses the train split by default; pass --combine to also fold in val.
    # --training-fraction / --val-fraction are ignored in --final mode.
    if fmt == "webdataset":
        try:
            with open(Path(data_root) / "wds" / "dataset_info.json") as _f:
                _meta = json.load(_f)
            _has_val = "val" in _meta.get("splits", {})
        except Exception:
            _has_val = False
    else:
        _has_val = (data_path / "val").exists()

    combine = getattr(args, "combine", False)
    if combine and not _has_val:
        logger.error("--combine specified but no val split found")
        sys.exit(1)

    if combine and _has_val:
        logger.info("Final training: combining train + val splits (--combine)")
        train_loader, inferred_classes = _build_combined_loader(
            data_root, fmt, batch_size, args.workers, args.seed, train_tf, collate_fn)
    elif fmt == "webdataset":
        if _has_val:
            logger.info("Final training: using train split only (pass --combine to include val)")
        train_loader, _, inferred_classes = _build_wds_loaders(
            data_root, batch_size, args.workers, args.seed,
            train_tf, val_tf, collate_fn,
            training_fraction=1.0, val_fraction=None, require_val=False)
    else:
        if _has_val:
            logger.info("Final training: using train split only (pass --combine to include val)")
        train_loader, _, inferred_classes = _build_loaders(
            data_path, batch_size, args.workers, args.seed,
            train_tf, val_tf, collate_fn,
            training_fraction=1.0, val_fraction=None, require_val=False)
    if num_classes is None:
        num_classes = inferred_classes

    model = create_model(model_name, num_classes, args.pretrained, params.get("drop_rate", 0.0))
    model = model.to(device)

    if args.freeze_backbone > 0:
        freeze_backbone(model)

    optimizer = get_optimizer(model, params.get("optimizer", "AdamW"), params["lr"], params.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))

    if args.amp:
        _amp_dtype = get_amp_dtype()
        _amp_label = "BF16" if _amp_dtype == torch.bfloat16 else "FP16"
        logger.info(f"AMP enabled: {_amp_label}")

    stats_lines = []

    for epoch in range(epochs):
        if args.freeze_backbone > 0 and epoch == args.freeze_backbone:
            unfreeze_all(model)
            optimizer = get_optimizer(model, params.get("optimizer", "AdamW"), params["lr"], params.get("weight_decay", 1e-4))
            scheduler = build_scheduler(
                optimizer, epochs, len(train_loader),
                start_step=epoch * len(train_loader),
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_soft_labels=use_mixup_cutmix, epoch=epoch, epochs=epochs,
            use_amp=args.amp, show_progress=True
        )
        line = f"INFO: Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
        tqdm.write(line)
        stats_lines.append(line)

    summary_lines = ["", "Final training complete."]
    summary_lines.append(f"  Trained for {epochs} epochs on all available training data.")

    save_checkpoint = bool(getattr(args, "checkpoint", False))
    if save_checkpoint:
        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoint_path = args.final_model
        torch.save({
            "model_state_dict": final_state,
            "epoch": epochs,
            "params": params,
            "model_name": model_name,
            "num_classes": num_classes,
        }, checkpoint_path)
        summary_lines.append(f"  Checkpoint saved to: {checkpoint_path}")
    else:
        summary_lines.append("  Checkpoint not saved (pass --checkpoint to write .pt).")

    test_loader = _build_test_loader(args.test_data, fmt, batch_size, args.workers, args.seed, val_tf)
    if test_loader is not None:
        val_criterion = nn.CrossEntropyLoss()
        _, test_acc, test_auroc = evaluate(model, test_loader, val_criterion, device, use_amp=args.amp)
        summary_lines.append(f"  Test accuracy:      {test_acc:.4f}  test AUROC: {test_auroc:.4f}")
        class_names = _get_class_names(args.test_data, fmt, num_classes)
        probs_path = f"{args.prefix}_probs.csv"
        _save_probs_csv(model, test_loader, class_names, probs_path, device, use_amp=args.amp)
        summary_lines.append(f"  Test probs saved to: {probs_path}")
    else:
        summary_lines.append("  No --test-data provided — skipping test evaluation.")
    summary_lines.append(f"  Final-train time: {ck.format_duration(time.time() - t0)}")

    for line in summary_lines:
        tqdm.write(line)

    if args.final_stats:
        with open(args.final_stats, "w") as f:
            f.write("\n".join(stats_lines + summary_lines) + "\n")
        tqdm.write(f"  Stats written to: {args.final_stats}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(args):
    import numpy as np
    import tempfile
    from PIL import Image

    logger.info("Running smoke test...")
    _ck().set_seed(0)  # deterministic synthetic data so smoke_probs.csv is reproducible
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for split in ["train", "val"]:
            for cls in ["cat", "dog", "bird"]:
                cls_dir = tmpdir / split / cls
                cls_dir.mkdir(parents=True)
                for i in range(10):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    )
                    img.save(cls_dir / f"img_{i}.jpg")

        smoke_args = argparse.Namespace(
            data=str(tmpdir),
            model="resnet18",
            pretrained=False,
            n_trials=2,
            epochs=2,
            batch_size=4,
            output=str(tmpdir / "smoke_results.json"),
            seed=0,
            device=args.device,
            cpu=False,
            repro=False,
            workers=0,
            img_size=64,
            freeze_backbone=0,
            training_fraction=1.0,
            val_fraction=1.0,
            resume=None,
            search_space=None,
            final=None,
            smoke_test=False,
            num_train_workers=1,
            ray_address=None,
            ray_storage=None,
            tune_metric="val_auroc",
            combine=False,
            amp=False,
            checkpoint=True,
            final_model="tunic_final.pt",
            final_stats=None,
            shuffle=None,
            prefix="smoke",
            test_data=None,
            sampler="tpe",
            scheduler="asha",
            optuna_storage=None,
        )

        run_tuning(smoke_args)

        # Final mode
        smoke_final_args = argparse.Namespace(**vars(smoke_args))
        smoke_final_args.final = f"{smoke_args.prefix}.json"
        smoke_final_args.epochs = 2
        run_final(smoke_final_args)

    logger.info("Smoke test passed.")


# Preflight check
# ---------------------------------------------------------------------------

def _preflight_check_distribution(data_format, data_key, num_classes,
                                   training_fraction, val_fraction, seed):
    """Read train and val labels (no image loading) and exit on bad distribution."""
    import numpy as np

    if data_format == "webdataset":
        if data_key.startswith("s3://"):
            return  # skip preflight for remote data
        try:
            import webdataset as wds
        except ImportError:
            return

        with open(Path(data_key) / "wds" / "dataset_info.json") as f:
            meta = json.load(f)
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        def _decode_cls(key, data):
            return data if key.endswith(".cls") else None

        def _resolve_label(raw: bytes, c2i: dict) -> int:
            """Handle both integer labels (b'0') and string labels (b'atelectasis')."""
            s = raw.decode().strip()
            try:
                idx = int(s)
                if 0 <= idx < len(c2i):
                    return idx
            except ValueError:
                pass
            return c2i[s]

        def _read_labels(split, fraction):
            n_shards = meta["splits"][split]["num_shards"]
            n_total  = meta["splits"][split]["num_samples"]
            urls = [str(Path(data_key) / "wds" / split / f"shard-{i:06d}.tar")
                    for i in range(n_shards)]
            n_take = max(1, int(n_total * fraction))
            ds = (wds.WebDataset(urls, shardshuffle=False, empty_check=False)
                  .decode(_decode_cls)
                  .to_tuple("cls")
                  .map(lambda t: _resolve_label(t[0], class_to_idx)))
            all_labels = np.fromiter(ds, dtype=np.int64)
            return all_labels[:n_take], all_labels, n_total

        train_labels, train_all, n_train_total = _read_labels("train", training_fraction)
        class_names  = classes

        if "val" in meta["splits"]:
            vf = val_fraction if val_fraction is not None else 1.0
            val_labels, val_all, n_val_total = _read_labels("val", vf)
        else:
            # No val split: both fractions from training data, disjoint
            import random
            if val_fraction is None:
                print("\nError: No validation split found and --val-fraction is not set.", file=sys.stderr)
                print("Specify --val-fraction to reserve a fraction of training data for validation "
                      "(e.g. --val-fraction 0.2).", file=sys.stderr)
                print("--training-fraction + --val-fraction must be ≤ 1.0.", file=sys.stderr)
                sys.exit(1)
            n_total = len(train_all)
            if training_fraction + val_fraction > 1.0:
                print(f"\nError: --training-fraction ({training_fraction:.3f}) + "
                      f"--val-fraction ({val_fraction:.3f}) = {training_fraction + val_fraction:.3f} > 1.0",
                      file=sys.stderr)
                print("When there is no validation split, both fractions are drawn from the same training data "
                      "and must sum to ≤ 1.0.", file=sys.stderr)
                print("Reduce --training-fraction, --val-fraction, or both.", file=sys.stderr)
                sys.exit(1)

            # Use the same per-class stratified split the trainer uses,
            # so the validator's verdict matches what training will actually do.
            from collections import defaultdict
            class_to_idxs = defaultdict(list)
            for i, lbl in enumerate(train_all):
                class_to_idxs[int(lbl)].append(i)
            train_sel, val_sel, _ = _split_by_label(
                class_to_idxs, training_fraction, val_fraction, seed)
            train_labels = train_all[train_sel]
            val_labels   = train_all[val_sel]
            val_all      = train_all
            n_train_total = n_total
            n_val_total   = n_total

    else:  # imagefolder
        from torchvision import datasets as tvd
        import random
        data_path = Path(data_key)

        val_dir = data_path / "val"
        if val_dir.exists():
            eff_val_frac = val_fraction if val_fraction is not None else 1.0

            train_full = np.array(tvd.ImageFolder(str(data_path / "train")).targets, dtype=np.int64)
            n_train_total = len(train_full)
            if training_fraction < 1.0:
                idx = list(range(len(train_full)))
                random.Random(seed).shuffle(idx)
                train_labels = train_full[idx[:max(1, round(len(idx) * training_fraction))]]
            else:
                train_labels = train_full
            train_all = train_full

            val_full = np.array(tvd.ImageFolder(str(val_dir)).targets, dtype=np.int64)
            n_val_total = len(val_full)
            if eff_val_frac < 1.0:
                idx = list(range(len(val_full)))
                random.Random(seed).shuffle(idx)
                val_labels = val_full[idx[:max(1, round(len(idx) * eff_val_frac))]]
            else:
                val_labels = val_full
            val_all = val_full
        else:
            # No val dir: both fractions from same training pool, disjoint stratified split
            if val_fraction is None:
                print("\nError: No val/ directory found and --val-fraction is not set.", file=sys.stderr)
                print("Specify --val-fraction to reserve a fraction of training images for validation "
                      "(e.g. --val-fraction 0.2).", file=sys.stderr)
                print("--training-fraction + --val-fraction must be ≤ 1.0.", file=sys.stderr)
                sys.exit(1)
            if training_fraction + val_fraction > 1.0:
                print(f"\nError: --training-fraction ({training_fraction:.3f}) + "
                      f"--val-fraction ({val_fraction:.3f}) = {training_fraction + val_fraction:.3f} > 1.0",
                      file=sys.stderr)
                print("When there is no val/ directory, both fractions are drawn from the same training set "
                      "and must sum to ≤ 1.0.", file=sys.stderr)
                print("Reduce --training-fraction, --val-fraction, or both.", file=sys.stderr)
                sys.exit(1)

            from collections import defaultdict
            base_ds = tvd.ImageFolder(str(data_path / "train"))
            class_to_idxs = defaultdict(list)
            for i, (_, lbl) in enumerate(base_ds.samples):
                class_to_idxs[lbl].append(i)

            train_sel, val_sel = [], []
            rng = random.Random(seed)
            for lbl in sorted(class_to_idxs):
                idxs = list(class_to_idxs[lbl])
                rng.shuffle(idxs)
                n_tr = max(1, round(len(idxs) * training_fraction))
                n_va = max(1, round(len(idxs) * val_fraction))
                train_sel.extend(idxs[:n_tr])
                val_sel.extend(idxs[n_tr:n_tr + n_va])

            targets = np.array(base_ds.targets, dtype=np.int64)
            n_train_total = len(targets)
            train_all = targets
            train_labels = targets[train_sel]

            n_val_total = len(targets)
            val_all = targets
            val_labels = targets[val_sel]

        class_names = None

    name_w = max((len(n) for n in class_names), default=7) if class_names else len(f"class {num_classes - 1:3d}")
    sel_w  = max(len(str(max((train_labels == c).sum() for c in range(num_classes)))),
                 len(str(max((val_labels   == c).sum() for c in range(num_classes)))))
    tot_w  = max(len(str(max((train_all   == c).sum() for c in range(num_classes)))),
                 len(str(max((val_all     == c).sum() for c in range(num_classes)))))

    train_bad = _report_split("Training",   train_labels, train_all, n_train_total, training_fraction,
                              num_classes, class_names, name_w, sel_w, tot_w, check_auroc=False)
    print()
    val_bad   = _report_split("Validation", val_labels,   val_all,   n_val_total,   val_fraction,
                              num_classes, class_names, name_w, sel_w, tot_w, check_auroc=True)

    errors = []
    if train_bad:
        errors.append(f"{len(train_bad)} training class(es) have no images - increase --training-fraction.")
    if val_bad:
        errors.append(f"{len(val_bad)} validation class(es) have no images - increase --val-fraction.")
    if errors:
        print()
        for msg in errors:
            print(msg)
        sys.exit(1)


def _report_split(split_name, labels, all_labels, n_total, fraction,
                  num_classes, class_names, name_w, sel_w, tot_w, check_auroc):
    """Print aligned per-class counts for one split. Return list of bad class indices."""
    import numpy as np
    n = len(labels)
    if fraction is not None and fraction < 1.0:
        header = f"{split_name} set: {n} images selected out of {n_total}, {num_classes} classes:"
    else:
        header = f"{split_name} set: {n_total} images, {num_classes} classes:"
    bad = []
    print(header)
    for c in range(num_classes):
        count     = int((labels     == c).sum())
        count_all = int((all_labels == c).sum())
        label = class_names[c] if class_names else f"class {c:3d}"
        flagged = (count == 0 or count == n) if check_auroc else (count == 0)
        note = "  <- no positives - AUROC undefined" if (check_auroc and flagged) else \
               "  <- no images" if flagged else ""
        if flagged:
            bad.append(c)
        if fraction is not None and fraction < 1.0:
            count_str = f"sampled {count:>{sel_w}} out of {count_all:>{tot_w}} images"
        else:
            count_str = f"{count_all:>{tot_w}} images"
        print(f"  {label:<{name_w}}  {count_str}{note}")
    return bad


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def run_tuning(args):
    import torch
    try:
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune import RunConfig
        _ray_available = True
    except ImportError:
        _ray_available = False
    from torchvision import datasets

    # Handle --cpu and --repro flags
    if hasattr(args, 'cpu') and args.cpu:
        args.device = "cpu"
        logger.info("Forcing CPU-only mode")

    if hasattr(args, 'repro') and args.repro:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Reproducibility mode enabled (may reduce GPU utilization)")
    elif hasattr(args, 'repro'):  # repro flag exists but is False
        torch.backends.cudnn.benchmark = True
        logger.info("Performance mode enabled (results may vary slightly across runs)")

    ck = _ck()
    set_seed = ck.set_seed
    validate_dataset_path = ck.validate_dataset_path
    get_amp_dtype = ck.get_amp_dtype
    load_search_space_overrides = ck.load_search_space_overrides

    if not args.model:
        logger.error("--model is required for tuning. E.g. --model resnet50")
        sys.exit(1)

    if not _ray_available:
        logger.error("Ray is not installed. Install with: pip install 'ray[tune,train]' optuna")
        sys.exit(1)

    # Dataset detection and preflight checks must run before Ray starts.
    data_root   = args.data
    data_format = _detect_format(data_root)
    set_seed(args.seed)

    if data_format == "webdataset":
        if not data_root.startswith("s3://"):
            validate_dataset_path(Path(data_root))
        with open(Path(data_root) / "wds" / "dataset_info.json") as f:
            wds_meta = json.load(f)
        num_classes = len(wds_meta["classes"])
        data_key = data_root
        n_train_total = wds_meta.get("splits", {}).get("train", {}).get("num_samples")
    else:
        data_path = Path(data_root)
        validate_dataset_path(data_path)
        tmp_ds = datasets.ImageFolder(str(data_path / "train"))
        num_classes = len(tmp_ds.classes)
        data_key = str(data_path.resolve())
        n_train_total = len(tmp_ds.samples)

    n_train_est = int(n_train_total * args.training_fraction) if n_train_total else None

    logger.info(f"Dataset: {data_root} | Format: {data_format} | Classes: {num_classes} | Model: {args.model}")
    _preflight_check_distribution(data_format, data_key, num_classes,
                                  args.training_fraction, args.val_fraction, args.seed)

    if not ray.is_initialized():
        address = args.ray_address or None
        ray.init(address=address, ignore_reinit_error=True)
        logger.info(f"Ray initialized (address={address or 'local'})")

    if args.amp:
        _amp_dtype = get_amp_dtype()
        _amp_label = "BF16" if _amp_dtype == torch.bfloat16 else "FP16"
        logger.info(f"AMP enabled: {_amp_label}")

    ss = {}
    if args.search_space:
        ss = load_search_space_overrides(args.search_space)

    # Load normalization params from dataset_info.json if available
    norm_params = _load_normalization_params(data_key)
    if norm_params:
        logger.info(f"Using dataset normalization: pool_mean={norm_params.get('pool_mean'):.2f}, "
                   f"pool_std={norm_params.get('pool_std'):.2f}")

    hp_keys = ["lr", "weight_decay", "label_smoothing", "drop_rate",
               "randaugment_magnitude", "randaugment_num_ops",
               "mixup_alpha", "cutmix_alpha", "optimizer", "batch_size"]

    use_gpu = _ck().get_device(args.device).type == "cuda"

    # Determine batch_size: fixed if explicitly provided, otherwise tune.
    # Filter candidates so each yields >= 1 full batch (drop_last=True), else
    # train_one_epoch divides by zero.
    bs_candidates = [16, 32, 64, 128]
    if n_train_est is not None:
        kept = [bs for bs in bs_candidates if bs <= n_train_est]
        if not kept:
            kept = [max(1, n_train_est)]
            logger.warning(f"Only {n_train_est} training samples; using batch_size={kept[0]}")
        elif kept != bs_candidates:
            dropped = [bs for bs in bs_candidates if bs not in kept]
            logger.warning(f"Dropping batch_size {dropped} (exceeds {n_train_est} train samples)")
        bs_candidates = kept

    if args.batch_size is None:
        if len(bs_candidates) == 1:
            batch_size_value = bs_candidates[0]
            logger.info(f"Batch size fixed to {batch_size_value} (only viable choice)")
        else:
            batch_size_value = tune.choice(bs_candidates)
            logger.info(f"Batch size will be tuned over {bs_candidates}")
    else:
        if n_train_est is not None and args.batch_size > n_train_est:
            logger.error(
                f"--batch-size {args.batch_size} exceeds available training samples "
                f"({n_train_est}); training would produce 0 full batches."
            )
            sys.exit(1)
        batch_size_value = args.batch_size
        logger.info(f"Batch size fixed to {args.batch_size}")

    search_space = {
        "data": data_key,
        "data_format": data_format,
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": batch_size_value,
        "img_size": args.img_size,
        "freeze_backbone": args.freeze_backbone,
        "seed": args.seed,
        "dataloader_workers": args.workers,
        "training_fraction": args.training_fraction,
        "val_fraction": args.val_fraction,
        "num_classes": num_classes,
        "device": args.device,
        "use_amp": args.amp,
        "n_trials": args.n_trials,
        "shuffle_seed": args.shuffle,
        "normalization": norm_params,
        "lr":                    tune.loguniform(ss.get("lr_min", 1e-5),    ss.get("lr_max", 1e-1)),
        "weight_decay":          tune.loguniform(ss.get("wd_min", 1e-6),    ss.get("wd_max", 1e-2)),
        "label_smoothing":       tune.uniform(   ss.get("ls_min", 0.0),     ss.get("ls_max", 0.2)),
        "drop_rate":             tune.uniform(   ss.get("dr_min", 0.0),     ss.get("dr_max", 0.5)),
        "randaugment_magnitude": tune.randint(   ss.get("ra_mag_min", 0),   ss.get("ra_mag_max", 15) + 1),
        "randaugment_num_ops":   tune.randint(   ss.get("ra_ops_min", 1),   ss.get("ra_ops_max", 3) + 1),
        "mixup_alpha":           tune.uniform(   ss.get("mixup_min", 0.0),  ss.get("mixup_max", 0.4)),
        "cutmix_alpha":          tune.uniform(   ss.get("cutmix_min", 0.0), ss.get("cutmix_max", 1.0)),
        "optimizer":             tune.choice(    ss.get("optimizers",       ["AdamW", "SGD"])),
    }

    trainable = tune.with_resources(
        _tune_trial,
        resources={"GPU": 1 if use_gpu else 0, "CPU": max(1, args.workers)},
    )

    import optuna
    _optuna_sampler = (optuna.samplers.RandomSampler(seed=args.seed)
                       if args.sampler == "random" else None)  # None -> Optuna default TPE
    _optuna_kwargs = {"metric": args.tune_metric, "mode": "max"}
    if _optuna_sampler is not None:
        _optuna_kwargs["sampler"] = _optuna_sampler  # sampler carries its own seed
    else:
        _optuna_kwargs["seed"] = args.seed  # default TPE seeded via top-level kwarg
    if args.optuna_storage:
        _optuna_kwargs["storage"] = optuna.storages.RDBStorage(url=args.optuna_storage)
        _optuna_kwargs["study_name"] = args.prefix
    search_alg = OptunaSearch(**_optuna_kwargs)
    logger.info(f"Sampler: {args.sampler}  Scheduler: {args.scheduler}"
                + (f"  Storage: {args.optuna_storage}" if args.optuna_storage else ""))

    if args.scheduler == "none":
        scheduler = None
    else:
        scheduler = ASHAScheduler(
            metric=args.tune_metric,
            mode="max",
            max_t=args.epochs,
            grace_period=max(1, args.epochs // 5),
        )

    if args.ray_storage:
        storage_path = args.ray_storage
    else:
        storage_path = str(Path(f"{args.prefix}.json").parent.resolve() / "ray_results")

    class _IntermediateResultsCallback(tune.Callback):
        def __init__(self):
            self.completed = []

        def on_trial_complete(self, iteration, trials, trial, **kwargs):
            if not trial.last_result:
                return
            self.completed.append({
                "val_acc":   trial.last_result.get("val_acc"),
                "val_auroc": trial.last_result.get("val_auroc"),
                "params":    {k: trial.config[k] for k in hp_keys if k in trial.config},
                "status":    trial.status,
            })
            valid = [t for t in self.completed if t.get(args.tune_metric) is not None]
            if not valid:
                return
            best = max(valid, key=lambda t: t.get(args.tune_metric, float("-inf")))
            snapshot = {
                "best_val_acc":     best.get("val_acc"),
                "best_val_auroc":   best.get("val_auroc"),
                "best_params":      best.get("params", {}),
                "completed_trials": len(self.completed),
                "model":            args.model,
                "epochs":           args.epochs,
                "all_trials":       self.completed,
                "status":           "in_progress",
            }
            with open(f"{args.prefix}.json", "w") as f:
                json.dump(snapshot, f, indent=2)

    run_config = RunConfig(
        storage_path=storage_path,
        name="tunic_study",
        callbacks=[_IntermediateResultsCallback()],
    )

    if args.resume:
        resume_path = str(Path(args.resume).resolve())
        logger.info(f"Loading previous results from {resume_path} to warm-start search")
        prior_grid = tune.ExperimentAnalysis(resume_path)
        points_to_evaluate, evaluated_rewards = [], []
        for trial in prior_grid.trials:
            if trial.last_result and trial.status == "TERMINATED":
                cfg = {k: trial.config[k] for k in hp_keys if k in trial.config}
                acc = trial.last_result.get("val_acc")
                if cfg and acc is not None:
                    points_to_evaluate.append(cfg)
                    evaluated_rewards.append(acc)
        logger.info(f"Warm-starting from {len(points_to_evaluate)} prior trials, running {args.n_trials} new trials")
        _resume_kwargs = dict(_optuna_kwargs)
        _resume_kwargs["metric"] = "val_acc"
        _resume_kwargs["points_to_evaluate"] = points_to_evaluate
        _resume_kwargs["evaluated_rewards"] = evaluated_rewards
        search_alg = OptunaSearch(**_resume_kwargs)

    @ray.remote
    class TrialCounter:
        def __init__(self): self._n = 0
        def next(self): self._n += 1; return self._n

    TrialCounter.options(name="trial_counter", lifetime="detached", get_if_exists=True).remote()

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.n_trials,
        ),
        run_config=run_config,
    )

    start_time = time.time()
    try:
        results = tuner.fit()
    except KeyboardInterrupt:
        logger.info("Interrupted — saving current results...")
        raise
    total_time = time.time() - start_time

    best = results.get_best_result(metric=args.tune_metric, mode="max")
    best_val_acc = best.metrics["val_acc"]
    best_val_auroc = best.metrics.get("val_auroc", float("nan"))
    best_params = {k: best.config[k] for k in hp_keys if k in best.config}

    all_trials = []
    completed = 0
    errored = 0
    for r in results:
        state = "ERROR" if r.error else "COMPLETE"
        if r.error:
            errored += 1
        else:
            completed += 1
        all_trials.append({
            "val_acc": r.metrics.get("val_acc") if r.metrics else None,
            "val_auroc": r.metrics.get("val_auroc") if r.metrics else None,
            "params": {k: r.config[k] for k in hp_keys if k in r.config} if r.config else {},
            "state": state,
        })

    output = {
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "best_params": best_params,
        "model": args.model,
        "dataset": data_key,
        "num_classes": num_classes,
        "n_trials": args.n_trials,
        "epochs": args.epochs,
        "completed_trials": completed,
        "errored_trials": errored,
        "total_time_seconds": total_time,
        "all_trials": all_trials,
    }

    with open(f"{args.prefix}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}  AUROC: {best_val_auroc:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {args.prefix}.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    from importlib.metadata import version as _version, PackageNotFoundError
    try:
        _ver = _version("cvic")
    except PackageNotFoundError:
        _ver = "dev"
    class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    epilog = (
        "Hyperparameter search space (per trial; override via --search-space YAML):\n"
        "  batch_size             choice    [16, 32, 64, 128]\n"
        "  lr                     loguniform[1e-5, 1e-1]\n"
        "  weight_decay           loguniform[1e-6, 1e-2]\n"
        "  label_smoothing        uniform   [0.0, 0.2]\n"
        "  drop_rate              uniform   [0.0, 0.5]\n"
        "  randaugment_magnitude  randint   [0, 15]\n"
        "  randaugment_num_ops    randint   [1, 3]\n"
        "  mixup_alpha            uniform   [0.0, 0.4]\n"
        "  cutmix_alpha           uniform   [0.0, 1.0]\n"
        "  optimizer              choice    [AdamW, SGD]\n"
    )

    p = argparse.ArgumentParser(
        description=f"tunic {_ver} — hyperparameter tuning for image classifiers",
        formatter_class=_Fmt,
        epilog=epilog,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_ver}")
    p.add_argument("--data", type=str, help="Path to dataset root (ImageFolder layout)")
    p.add_argument("--test-data", type=str, default=None, dest="test_data",
                   help="Path that *is* the test data. Two accepted layouts: "
                        "(1) WebDataset: <test-data>/wds/dataset_info.json with 'test' in splits; "
                        "(2) ImageFolder: <test-data> itself contains class subdirs. "
                        "If omitted, no test evaluation happens. Never inferred from --data.")
    p.add_argument("--model", type=str, default=None,
                   help="Any timm model name (e.g. resnet50, efficientnet_b0, convnext_tiny)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Use timm pretrained weights")
    p.add_argument("--n-trials", type=int, default=80, dest="n_trials",
                   help="Number of Optuna trials")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs per trial")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="Batch size (default: tune over 16, 32, 64, 128; if specified, use as fixed value)")
    p.add_argument("--prefix", type=str, default="tunic",
                   help="Prefix for output files (e.g. --prefix myrun → myrun.json)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--random-seed", action="store_true", dest="random_seed",
                   help="Use a random seed derived from wallclock time (ignores --seed)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device to use: auto detects CUDA/MPS/CPU, or specify explicitly")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only mode (useful for reproducible results)")
    p.add_argument("--repro", action="store_true",
                   help="Enable reproducibility mode (deterministic algorithms, no cuDNN benchmark); trades performance for exact reproducibility")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count")
    p.add_argument("--img-size", type=int, default=224, dest="img_size",
                   help="Input image resolution")
    p.add_argument("--training-fraction", type=float, default=1.0, dest="training_fraction",
                   help="Fraction of training data to use (e.g. 0.1 for 10%%); same subset across all trials")
    p.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                   help="Fraction of data for validation. With a val/ dir: fraction of val split to use. "
                        "Without val/ dir: REQUIRED — fraction of total training images reserved for "
                        "validation (disjoint from training set); "
                        "--training-fraction + --val-fraction must be ≤ 1.0.")
    p.add_argument("--freeze-backbone", type=int, default=0, dest="freeze_backbone",
                   help="Epochs to freeze backbone; 0 = no freeze")
    p.add_argument("--final", type=str, default=None,
                   help="Path to tunic_results.json — skip tuning, train final model")
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=None,
                   help="Save the final model checkpoint (.pt). Default: True in --final mode, False after tuning. "
                        "Test evaluation runs whenever --test-data is provided, regardless of this flag.")
    p.add_argument("--no-final", action="store_true", dest="no_final",
                   help="Skip final-model training after HPO (overrides --checkpoint and --test-data triggers); "
                        "useful when you only want the HPO leaderboard.")
    p.add_argument("--combine", action="store_true", default=False,
                   help="When training the final model, fold the val split into training (default: train split only; exits if no val split)")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable automatic mixed precision (AMP) for faster training on CUDA (applies to both tuning trials and final training)")
    p.add_argument("--final-model", type=str, default="tunic_final.pt", dest="final_model",
                   help="Output filename for the final model checkpoint (default: tunic_final.pt)")
    p.add_argument("--final-stats", type=str, default=None, dest="final_stats",
                   help="Output filename for final training stats text file (optional)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a previous Ray Tune experiment directory; warm-starts Optuna search from those results and runs --n-trials new trials")
    p.add_argument("--search-space", type=str, default=None, dest="search_space",
                   help="YAML file to override search space ranges")
    p.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                   help="Run end-to-end smoke test with synthetic data")
    p.add_argument("--num-train-workers", type=int, default=1, dest="num_train_workers",
                   help="Ray Train workers per trial (= GPUs per trial)")
    p.add_argument("--ray-address", type=str, default=None, dest="ray_address",
                   help="Ray cluster address to connect to (e.g. localhost:6385); defaults to local")
    p.add_argument("--ray-storage", type=str, default=None, dest="ray_storage",
                   help="Ray Tune storage path (local dir or S3 URI, e.g. s3://bucket/ray-results)")
    p.add_argument("--tune-metric", type=str, default="val_auroc", dest="tune_metric",
                   help="Metric used by Optuna and ASHA for trial selection and pruning (default: val_auroc)")
    p.add_argument("--shuffle", type=int, default=None, metavar = 'shuffle_seed',
                   help="If set, each trial gets a unique T/V split derived from shuffle_seed + trial_number; "
                        "omit for a fixed split shared across all trials")
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"], dest="sampler",
                   help="Optuna sampler: 'tpe' (default, adaptive) or 'random' (uniform; use for HP-importance "
                        "studies — fANOVA needs uniform coverage of the search space)")
    p.add_argument("--scheduler", type=str, default="asha", choices=["asha", "none"], dest="scheduler",
                   help="Trial scheduler: 'asha' (default, early-stop bad trials) or 'none' (run every trial to "
                        "full --epochs; use for HP-importance studies to avoid early-stop confounds)")
    p.add_argument("--optuna-storage", type=str, default=None, dest="optuna_storage",
                   help="Optuna storage URL (e.g. sqlite:///study.db) to persist the study DB for post-hoc fANOVA "
                        "importance analysis. Omit for in-memory only.")
    return p.parse_args()


def main():
    _t_program_start = time.time()
    try:
        _main_body()
    finally:
        print(f"Total time: {_ck().format_duration(time.time() - _t_program_start)}")


def _main_body():
    args = parse_args()

    if args.cpu:
        args.device = "cpu"
        logger.info("Forcing CPU-only mode")

    if args.random_seed:
        args.seed = int(time.time() * 1e6) % (2**31)
        logger.info(f"Generated random seed: {args.seed}")

    # Set reproducibility or performance mode
    import torch
    if args.repro:
        # Reproducibility mode: sacrifice performance for exact reproducibility
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Reproducibility mode enabled (may reduce GPU utilization)")
    else:
        # Performance mode: optimize for speed
        torch.backends.cudnn.benchmark = True
        logger.info("Performance mode enabled (results may vary slightly across runs)")

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.final:
        if args.checkpoint is None:
            args.checkpoint = True
        run_final(args)
        return

    if not args.data:
        logger.error("--data is required for tuning mode")
        sys.exit(1)

    t0 = time.time()
    run_tuning(args)
    logger.info(f"HPO time: {_ck().format_duration(time.time() - t0)}")

    if args.checkpoint is None:
        args.checkpoint = False

    if args.no_final:
        logger.info("Skipping final model training (--no-final).")
        return

    has_test = _has_test_split(args.test_data)
    if not args.checkpoint and not has_test:
        logger.info("Skipping final model training (no --checkpoint and no --test-data).")
        return

    if args.checkpoint:
        logger.info("Training final model on all available data...")
    else:
        logger.info("Training final model to evaluate on test set (--no-checkpoint: .pt will not be saved).")
    try:
        import ray
        ray.shutdown()
    except Exception:
        pass
    args.final = f"{args.prefix}.json"
    args.final_model = f"{args.prefix}.pt"
    run_final(args)


if __name__ == "__main__":
    main()
