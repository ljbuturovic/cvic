"""common_cvic.py — Shared utilities for tunic.py and cvic.py."""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("cvic")


# ---------------------------------------------------------------------------
# Seeds / device
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def format_duration(seconds: float) -> str:
    """Human-friendly duration: minutes once past 5 min, else seconds."""
    return f"{seconds / 60:.1f}m" if seconds > 300 else f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def validate_dataset_path(data_path: Path):
    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        sys.exit(1)
    if not (data_path / "train").exists() and not (data_path / "wds" / "train").exists():
        logger.error(f"Expected a 'train/' or 'wds/train/' subdirectory in {data_path}")
        sys.exit(1)


def make_stratified_split(dataset, val_fraction: float = 0.2, seed: int = 42):
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    rng = random.Random(seed)
    for label, indices in class_to_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        split = max(1, int(len(indices) * val_fraction))
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int, randaug_magnitude: int = 0, randaug_num_ops: int = 2, is_train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        base = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug_magnitude > 0:
            base.append(RandAugment(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
        base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(base)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# In-memory cached dataset (decode once, augment fresh every epoch)
# ---------------------------------------------------------------------------

class CachedImageDataset(torch.utils.data.Dataset):
    """In-memory dataset of pre-decoded PIL images with the transform applied lazily.

    Images are decoded ONCE (the expensive JPEG/PNG decode + I/O) and held in RAM.
    The transform runs inside ``__getitem__``, so any randomness it contains
    (RandomResizedCrop, RandomHorizontalFlip, RandAugment) is re-sampled on *every*
    access — i.e. each image gets a fresh augmentation every epoch, which is the
    standard training regime. Pass a deterministic (``is_train=False``) transform for
    validation/test.

    Two views over the same images (e.g. an augmented train view and a clean val
    view) can share the same ``images`` list at no extra memory cost.

    NOTE: this is deliberately *not* a precompute-and-store-tensors cache. Caching the
    post-transform tensor would freeze the augmentation across epochs (every epoch
    sees the identical augmented image), which weakens regularization and makes a
    method that does it incomparable to one that augments fresh per epoch.
    """

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def load_wds_images(data_root: str, split: str = "train"):
    """Decode an entire WebDataset split into memory ONCE as decoded PIL images.

    Returns ``(images, labels, num_classes)`` where ``images`` is a list of RGB
    ``PIL.Image`` and ``labels`` is a ``list[int]``. No transform is applied — wrap
    the result in :class:`CachedImageDataset` so augmentation is re-sampled fresh
    every epoch. Supports local paths and ``s3://`` (streamed via ``aws s3 cp``).
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed. Run: pip install webdataset")
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

    classes = meta["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    available = list(meta["splits"].keys())
    if split not in available:
        logger.warning(f"Split '{split}' not found. Available: {available}. Using '{available[0]}'")
        split = available[0]

    def decode_cls(b):
        s = b.decode().strip()
        try:
            idx = int(s)
            if 0 <= idx < num_classes:
                return idx
        except ValueError:
            pass
        return class_to_idx[s]

    # webdataset >= 1.0 folds basichandlers into imagehandler, which maps ".cls" ->
    # int(data); intercept .cls to keep raw bytes so decode_cls handles string labels.
    _img_decoder = wds.autodecode.imagehandler("pil")

    def _decoder(key, data):
        if key.endswith(".cls"):
            return data
        return _img_decoder(key, data)

    n_shards = meta["splits"][split]["num_shards"]
    if is_s3:
        base = data_root.rstrip("/")
        urls = [f"pipe:aws s3 cp {base}/wds/{split}/shard-{i:06d}.tar -" for i in range(n_shards)]
    else:
        d = Path(data_root) / "wds" / split
        urls = [str(d / f"shard-{i:06d}.tar") for i in range(n_shards)]

    dataset = (
        wds.WebDataset(urls, shardshuffle=False, nodesplitter=wds.split_by_node, empty_check=False)
        .decode(_decoder)
        .to_tuple("png", "cls")
        .map_tuple(lambda img: img.convert("RGB"), decode_cls)
    )

    images, labels = [], []
    for img, lbl in dataset:
        images.append(img)
        labels.append(lbl)
    return images, labels, num_classes


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------

class MixupCutmixCollator:
    def __init__(self, mixup_alpha: float, cutmix_alpha: float, num_classes: int):
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes  = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = random.random() > 0.5
        elif self.cutmix_alpha > 0:
            use_cutmix = True
        elif self.mixup_alpha > 0:
            use_cutmix = False
        else:
            return images, labels

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            images, labels_a, labels_b = self._cutmix(images, labels, lam)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(images.size(0))
            images   = lam * images + (1 - lam) * images[idx]
            labels_a = labels
            labels_b = labels[idx]

        labels_a_oh  = nn.functional.one_hot(labels_a, self.num_classes).float()
        labels_b_oh  = nn.functional.one_hot(labels_b, self.num_classes).float()
        mixed_labels = lam * labels_a_oh + (1 - lam) * labels_b_oh
        return images, mixed_labels

    def _cutmix(self, images, labels, lam):
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        idx = torch.randperm(images.size(0))
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return images, labels, labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(model_name: str, num_classes: int, pretrained: bool, drop_rate: float) -> nn.Module:
    try:
        model = timm.create_model(model_name, pretrained=pretrained,
                                  num_classes=num_classes, drop_rate=drop_rate)
    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {e}")
        logger.error("Common alternatives: resnet50, efficientnet_b0, convnext_tiny, vit_small_patch16_224, mobilenetv3_large_100")
        sys.exit(1)
    return model


def freeze_backbone(model: nn.Module):
    head_keywords = {"head", "fc", "classifier"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top not in head_keywords and not any(kw in name for kw in head_keywords):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5,
                    start_step: int = 0):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training / evaluation primitives
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    use_soft_labels, trial_id="", epoch=0, epochs=0,
                    use_amp=False, show_progress=True):
    import sys
    import time
    model.train()
    use_amp    = use_amp and device.type == "cuda"
    amp_dtype  = get_amp_dtype()
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # Accumulate metrics on-device so there is no GPU->CPU sync per batch; we sync
    # only for the (throttled) progress display and once at epoch end. A per-batch
    # .item() forces the CPU to block on each step, which can throttle a fast GPU.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)  # float64 to match prior Python-float accumulation
    correct    = torch.zeros((), dtype=torch.long, device=device)
    total      = 0

    epoch_str = f" epoch {epoch+1}/{epochs}" if epochs else ""
    desc = f"trial {trial_id}{epoch_str}" if trial_id else f"train{epoch_str}"
    # Refresh fast on a terminal, but coarsely when redirected to a file/pipe —
    # otherwise log size scales with epoch wall-time (slower GPUs → huge logs).
    _mininterval = 0.1 if sys.stderr.isatty() else 10.0
    bar = tqdm(loader, leave=False, desc=desc, disable=not show_progress,
               mininterval=_mininterval,
               bar_format="{l_bar}{bar}| batch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    _can_postfix = hasattr(bar, "set_postfix")
    _last_postfix = 0.0

    for images, labels in bar:
        images = images.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            if use_soft_labels and labels.dim() == 2:
                labels = labels.to(device)
                loss = -(labels * nn.functional.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
            else:
                labels = labels.to(device)
                loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        target = labels.argmax(dim=1) if (use_soft_labels and labels.dim() == 2) else labels

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # On-device accumulation — no .item(), so no per-batch sync.
        bs = images.size(0)
        total_loss += loss.detach() * bs
        correct    += (preds == target).sum()
        total      += bs

        # Throttled live display: sync at most once per refresh interval.
        if _can_postfix:
            now = time.monotonic()
            if now - _last_postfix >= _mininterval:
                _last_postfix = now
                bar.set_postfix(loss=f"{(total_loss / total).item():.4f}",
                                acc=f"{(correct / total).item():.4f}")

    # Single sync for the returned epoch metrics.
    return (total_loss / total).item(), (correct / total).item()


def check_class_distribution(labels: np.ndarray, n_classes: int, class_names: list[str] | None = None) -> list[int]:
    """Print per-class sample counts to stderr. Return list of unscorable class indices.

    Does NOT call sys.exit() — the caller decides whether to abort.
    """
    n = len(labels)
    bad = []
    lines = [f"\nValidation set: {n} samples across {n_classes} classes:"]
    for c in range(n_classes):
        count = int((labels == c).sum())
        if count == 0:
            note = "  <- no positives — AUROC undefined"
            bad.append(c)
        elif count == n:
            note = "  <- no negatives — AUROC undefined"
            bad.append(c)
        else:
            note = ""
        label = f"{class_names[c]}" if class_names else f"class {c:3d}"
        lines.append(f"  {label:30s}: {count:5d} samples{note}" if class_names else f"  {label}: {count:5d} samples{note}")
    if bad:
        lines.append(f"\n{len(bad)} class(es) cannot be scored.")
        lines.append("Increase --val-fraction so every class has both positive and negative examples.")
        print("\n".join(lines), file=sys.stderr)
    return bad


def _compute_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    present = np.unique(labels)
    try:
        if probs.shape[1] == 2:
            return roc_auc_score(labels, probs[:, 1])
        if len(present) < 2:
            return float("nan")
        return roc_auc_score(labels, probs[:, present], multi_class="ovr",
                             average="macro", labels=present)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Search space overrides
# ---------------------------------------------------------------------------

def load_search_space_overrides(path: str) -> dict:
    if yaml is None:
        logger.error("PyYAML is required for --search-space. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}
