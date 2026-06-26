#!/usr/bin/env python3
"""cvic.py — Cross-validation hyperparameter search for image classifiers using Ray Tune + timm."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cvic")


def _ck():
    """Import shared utilities (cached by sys.modules after first call)."""
    try:
        import common_cvic as _m
    except ImportError:
        import cvic.common_cvic as _m
    return _m


def _detect_format(data_root: str) -> str:
    """Return 'webdataset' or 'imagefolder' based on what's present at data_root."""
    if data_root.startswith("s3://"):
        return "webdataset"
    p = Path(data_root)
    if (p / "wds" / "dataset_info.json").exists():
        return "webdataset"
    return "imagefolder"


def _uses_ray_gpu(device_str: str) -> bool:
    """Return whether Ray should reserve a CUDA GPU for this run."""
    return _ck().get_device(device_str).type == "cuda"


def _compute_metric(probs, labels, metric: str) -> float:
    import numpy as np
    ck = _ck()
    if "auroc" in metric:
        return ck._compute_auroc(probs, labels)
    if "acc" in metric:
        return float(np.mean(np.argmax(probs, axis=1) == labels))
    return ck._compute_auroc(probs, labels)


def _cvic_trial(config: dict):
    """CV trial: trains one model per fold, reports aggregated metric to Ray Tune."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    import ray
    from ray import tune
    from sklearn.model_selection import StratifiedKFold, KFold
    ck = _ck()
    set_seed        = ck.set_seed
    get_device      = ck.get_device
    get_amp_dtype   = ck.get_amp_dtype
    build_transforms = ck.build_transforms
    create_model    = ck.create_model
    freeze_backbone = ck.freeze_backbone
    unfreeze_all    = ck.unfreeze_all
    get_optimizer   = ck.get_optimizer
    build_scheduler = ck.build_scheduler
    train_one_epoch = ck.train_one_epoch
    MixupCutmixCollator = ck.MixupCutmixCollator

    data_path          = Path(config["data"])
    device             = get_device(config["device"])
    n_folds            = config["n_folds"]
    n_repeats          = config["n_repeats"]
    stratified         = config["stratified"]
    pooling            = config["pooling"]
    tune_metric        = config["tune_metric"]
    epochs             = config["epochs"]
    batch_size         = config["batch_size"]
    img_size           = config["img_size"]
    workers            = config["dataloader_workers"]
    base_seed          = config["seed"]
    num_classes        = config["num_classes"]
    lr                 = config["lr"]
    weight_decay       = config["weight_decay"]
    label_smoothing    = config["label_smoothing"]
    drop_rate          = config["drop_rate"]
    randaug_magnitude  = config["randaugment_magnitude"]
    randaug_num_ops    = config["randaugment_num_ops"]
    mixup_alpha        = config["mixup_alpha"]
    cutmix_alpha       = config["cutmix_alpha"]
    optimizer_name     = config["optimizer"]
    use_amp            = config.get("use_amp", False)
    freeze_bb          = config["freeze_backbone"]

    _counter = ray.get_actor("trial_counter")
    _trial_num = ray.get(_counter.next.remote())
    trial_id = f"{_trial_num}/{config['n_trials']}"

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(img_size, randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf   = build_transforms(img_size, is_train=False)

    # Detect format and load data
    fmt = _detect_format(str(data_path))
    if fmt == "webdataset":
        logger.info("Loading WebDataset format")
        # Decode images ONCE; apply the transform lazily so augmentation is re-sampled
        # fresh every epoch (see CachedImageDataset). The two views share one image list.
        raw_images, all_labels, _ = ck.load_wds_images(str(data_path))
        base_ds_aug   = ck.CachedImageDataset(raw_images, all_labels, train_tf)
        base_ds_clean = ck.CachedImageDataset(raw_images, all_labels, val_tf)
    else:
        # Two views of the same data: augmented for training, clean for evaluation
        train_dir = data_path / "train"
        base_ds_aug   = datasets.ImageFolder(str(train_dir), transform=train_tf)
        base_ds_clean = datasets.ImageFolder(str(train_dir), transform=val_tf)
        all_labels = [label for _, label in base_ds_aug.samples]

    N = len(base_ds_aug)

    # probs_accum[i] sums out-of-fold softmax probs for sample i across repeats
    probs_accum  = np.zeros((N, num_classes), dtype=np.float64)
    labels_arr   = np.array(all_labels)
    fold_aurocs  = []   # per-fold AUROC (non-pooling)
    fold_accs    = []   # per-fold accuracy (non-pooling)

    try:
        for repeat in range(n_repeats):
            repeat_seed = base_seed + repeat * 10000
            set_seed(repeat_seed)

            if stratified:
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(range(N), all_labels))
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(range(N)))

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_seed = repeat_seed + fold_idx
                set_seed(fold_seed)
                fold_label = f"{trial_id} repeat{repeat+1}/{n_repeats} fold{fold_idx+1}/{n_folds}"

                train_loader = DataLoader(
                    Subset(base_ds_aug, train_idx),
                    batch_size=batch_size, shuffle=True,
                    num_workers=workers, pin_memory=(device.type == "cuda"),
                    drop_last=True, collate_fn=collate_fn,
                )
                val_loader = DataLoader(
                    Subset(base_ds_clean, val_idx),
                    batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=(device.type == "cuda"),
                )

                model = create_model(config["model"], num_classes, config["pretrained"], drop_rate)
                model = model.to(device)
                if freeze_bb > 0:
                    freeze_backbone(model)

                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                scheduler = build_scheduler(optimizer, epochs, len(train_loader))
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

                for epoch in range(epochs):
                    if freeze_bb > 0 and epoch == freeze_bb:
                        unfreeze_all(model)
                        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                        scheduler = build_scheduler(
                            optimizer, epochs, len(train_loader),
                            start_step=epoch * len(train_loader),
                        )
                    train_one_epoch(
                        model, train_loader, optimizer, scheduler, criterion, device,
                        use_soft_labels=use_mixup_cutmix,
                        trial_id=fold_label, epoch=epoch, epochs=epochs,
                        use_amp=use_amp, show_progress=True,
                    )

                # Collect out-of-fold predictions
                amp_dtype = get_amp_dtype()
                model.eval()
                fold_probs, fold_labels = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                            enabled=use_amp and device.type == "cuda"):
                            outputs = model(images)
                        fold_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
                        fold_labels.extend(labels.cpu().numpy())

                fold_probs_np  = np.concatenate(fold_probs)
                fold_labels_np = np.array(fold_labels)

                if pooling:
                    probs_accum[val_idx] += fold_probs_np
                else:
                    fold_aurocs.append(_compute_metric(fold_probs_np, fold_labels_np, "auroc"))
                    fold_accs.append(_compute_metric(fold_probs_np, fold_labels_np, "acc"))

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if pooling:
            probs_final = probs_accum / n_repeats
            auroc = _compute_metric(probs_final, labels_arr, "auroc")
            acc   = _compute_metric(probs_final, labels_arr, "acc")
        else:
            auroc = float(np.nanmean(fold_aurocs))
            acc   = float(np.nanmean(fold_accs))
        metric_val = auroc if "auroc" in tune_metric else acc

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metric_val = auroc = acc = 0.0

    # Report both genuine metrics so either can be read downstream.
    tune.report({tune_metric: metric_val, "val_auroc": auroc, "val_acc": acc})


def _evaluate_on_test_data(train_data_path: Path, test_data_path, best_params: dict, args):
    """Train on all training data with best hyperparams. Optionally evaluate on test
    data (if test_data_path is given) and/or save a model checkpoint (if args.checkpoint)."""
    import csv
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ck = _ck()
    set_seed        = ck.set_seed
    get_device      = ck.get_device
    get_amp_dtype   = ck.get_amp_dtype
    build_transforms = ck.build_transforms
    create_model    = ck.create_model
    freeze_backbone = ck.freeze_backbone
    unfreeze_all    = ck.unfreeze_all
    get_optimizer   = ck.get_optimizer
    build_scheduler = ck.build_scheduler
    train_one_epoch = ck.train_one_epoch
    MixupCutmixCollator = ck.MixupCutmixCollator

    set_seed(args.seed)
    device = get_device(args.device)

    train_tf = build_transforms(args.img_size, best_params.get("randaugment_magnitude", 0),
                                best_params.get("randaugment_num_ops", 1), is_train=True)
    val_tf   = build_transforms(args.img_size, is_train=False)

    # Detect format and load training data
    fmt = _detect_format(str(train_data_path))
    if fmt == "webdataset":
        logger.info("Loading WebDataset format for test evaluation")
        with open(train_data_path / "wds" / "dataset_info.json") as f:
            meta = json.load(f)
        class_names = meta["classes"]
        # Decode once; augment fresh per epoch via the lazy transform (CachedImageDataset).
        raw_images, train_labels, num_classes = ck.load_wds_images(str(train_data_path))
        train_ds_aug   = ck.CachedImageDataset(raw_images, train_labels, train_tf)
        train_ds_clean = ck.CachedImageDataset(raw_images, train_labels, val_tf)
    else:
        train_dir = train_data_path / "train"
        train_ds_aug   = datasets.ImageFolder(str(train_dir), transform=train_tf)
        train_ds_clean = datasets.ImageFolder(str(train_dir), transform=val_tf)
        num_classes = best_params.get("num_classes", len(train_ds_aug.classes))
        class_names = train_ds_aug.classes

    # Load test data (detect format) — skip if test_data_path is None (checkpoint-only mode).
    test_loader = None
    if test_data_path is not None:
        test_fmt = _detect_format(str(test_data_path))
        if test_fmt == "webdataset":
            # Test set: deterministic (clean) transform — no augmentation either way.
            test_imgs, test_labels, _ = ck.load_wds_images(str(test_data_path), split="test")
            test_ds = ck.CachedImageDataset(test_imgs, test_labels, val_tf)
        else:
            test_ds = datasets.ImageFolder(str(test_data_path), transform=val_tf)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=(device.type == "cuda"),
        )

    use_mixup_cutmix = best_params.get("mixup_alpha", 0) > 0 or best_params.get("cutmix_alpha", 0) > 0
    collate_fn = MixupCutmixCollator(best_params.get("mixup_alpha", 0),
                                     best_params.get("cutmix_alpha", 0), num_classes) if use_mixup_cutmix else None

    train_loader = DataLoader(
        train_ds_aug, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True, collate_fn=collate_fn,
    )

    # Train model on all data
    model = create_model(args.model, num_classes, args.pretrained, best_params.get("drop_rate", 0.0))
    model = model.to(device)
    if args.freeze_backbone > 0:
        freeze_backbone(model)

    epochs = best_params.get("epochs", args.epochs)
    optimizer = get_optimizer(model, best_params.get("optimizer", "AdamW"),
                            best_params.get("lr", 1e-3), best_params.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params.get("label_smoothing", 0.0))

    for epoch in range(epochs):
        if args.freeze_backbone > 0 and epoch == args.freeze_backbone:
            unfreeze_all(model)
            optimizer = get_optimizer(model, best_params.get("optimizer", "AdamW"),
                                    best_params.get("lr", 1e-3), best_params.get("weight_decay", 1e-4))
            scheduler = build_scheduler(
                optimizer, epochs, len(train_loader),
                start_step=epoch * len(train_loader),
            )
        train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_soft_labels=use_mixup_cutmix, trial_id="test_train", epoch=epoch, epochs=epochs,
            use_amp=args.amp, show_progress=True,
        )

    # Save model checkpoint if requested
    if getattr(args, "checkpoint", False):
        ckpt_path = f"{args.prefix}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "num_classes": num_classes,
            "class_names": class_names,
            "best_params": best_params,
        }, ckpt_path)
        logger.info(f"Checkpoint saved to: {ckpt_path}")

    # Evaluate on test data if test_loader was built
    if test_loader is None:
        return {}

    amp_dtype = get_amp_dtype()
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
            test_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs_np = np.concatenate(test_probs)
    test_labels_np = np.array(test_labels)

    test_acc = _compute_metric(test_probs_np, test_labels_np, "acc")
    test_auroc = _compute_metric(test_probs_np, test_labels_np, "auroc")

    # Save test probabilities to CSV (same format as tunic.py)
    csv_path = f"{args.prefix}_probs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sampleID", "truth"] + [f"prob_{c}" for c in class_names])
        for i, (lbl, probs) in enumerate(zip(test_labels_np, test_probs_np)):
            writer.writerow([i, class_names[int(lbl)]] + [f"{p:.6f}" for p in probs])
    logger.info(f"Test probabilities saved to: {csv_path}")

    return {"test_acc": test_acc, "test_auroc": test_auroc}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_cv(args):
    import os
    import numpy as np
    import torch
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune import RunConfig
    from torchvision import datasets
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Handle --cpu and --repro flags
    if args.cpu:
        args.device = "cpu"
        logger.info("Forcing CPU-only mode")

    if args.repro:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Reproducibility mode enabled (may reduce GPU utilization)")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info("Performance mode enabled (results may vary slightly across runs)")

    @ray.remote
    class TrialCounter:
        def __init__(self): self._n = 0
        def next(self): self._n += 1; return self._n

    ck = _ck()
    get_amp_dtype           = ck.get_amp_dtype
    validate_dataset_path   = ck.validate_dataset_path
    load_search_space_overrides = ck.load_search_space_overrides

    if not args.model:
        logger.error("--model is required. E.g. --model resnet18")
        sys.exit(1)

    data_root = args.data
    data_path = Path(data_root)
    validate_dataset_path(data_path)

    # Detect format and get num_classes
    fmt = _detect_format(data_root)
    if fmt == "webdataset":
        with open(data_path / "wds" / "dataset_info.json") as f:
            meta = json.load(f)
        num_classes = len(meta["classes"])
        N = meta["splits"]["train"]["num_samples"]
    else:
        train_dir = data_path / "train"
        tmp_ds = datasets.ImageFolder(str(train_dir))
        num_classes = len(tmp_ds.classes)
        N = len(tmp_ds)

    logger.info(
        f"Dataset: {data_root} ({fmt}) | Samples: {N} | Classes: {num_classes} | Model: {args.model}"
    )
    logger.info(
        f"CV: {args.folds} folds x {args.repeats} repeats | "
        f"{'stratified' if args.stratified else 'random'} | "
        f"{'pooling' if args.pooling else 'averaging'}"
    )
    if args.amp:
        amp_dtype = get_amp_dtype()
        logger.info(f"AMP enabled: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")

    use_gpu = _uses_ray_gpu(args.device)

    ss = {}
    if args.search_space:
        ss = load_search_space_overrides(args.search_space)

    hp_keys = ["epochs", "lr", "weight_decay", "label_smoothing", "drop_rate",
               "randaugment_magnitude", "randaugment_num_ops",
               "mixup_alpha", "cutmix_alpha", "optimizer"]

    epochs_min = ss.get("epochs_min", min(10, args.epochs))
    epochs_max = ss.get("epochs_max", args.epochs)
    if epochs_min > epochs_max:
        logger.error(
            f"epochs_min ({epochs_min}) > epochs_max ({epochs_max}); "
            "lower --epochs or set epochs_min in --search-space."
        )
        sys.exit(1)
    if epochs_min == epochs_max:
        epochs_value = epochs_min
        logger.info(f"Epochs fixed to {epochs_value} (min == max)")
    else:
        epochs_value = tune.randint(epochs_min, epochs_max + 1)
        logger.info(f"Epochs will be tuned over [{epochs_min}, {epochs_max}]")

    search_space = {
        "data":              data_root,
        "model":             args.model,
        "pretrained":        args.pretrained,
        "epochs":                epochs_value,
        "batch_size":        args.batch_size,
        "img_size":          args.img_size,
        "freeze_backbone":   args.freeze_backbone,
        "seed":              args.seed,
        "dataloader_workers": args.workers,
        "num_classes":       num_classes,
        "device":            args.device,
        "use_amp":           args.amp,
        "n_trials":          args.n_trials,
        "n_folds":           args.folds,
        "n_repeats":         args.repeats,
        "stratified":        args.stratified,
        "pooling":           args.pooling,
        "tune_metric":       args.tune_metric,
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
        _cvic_trial,
        resources={"GPU": 1 if use_gpu else 0, "CPU": max(1, args.workers)},
    )

    search_alg = OptunaSearch(metric=args.tune_metric, mode="max", seed=args.seed)

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
                args.tune_metric: trial.last_result.get(args.tune_metric),
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
                "epochs_max":       args.epochs,
                "n_folds":          args.folds,
                "n_repeats":        args.repeats,
                "all_trials":       self.completed,
                "status":           "in_progress",
            }
            with open(f"{args.prefix}.json", "w") as f:
                json.dump(snapshot, f, indent=2)

    run_config = RunConfig(
        storage_path=storage_path,
        name="cvic_study",
        callbacks=[_IntermediateResultsCallback()],
    )

    ray_address = getattr(args, "ray_address", None)
    ray.init(address=ray_address, ignore_reinit_error=True, namespace="cvic")
    logger.info(f"Ray initialized (address={ray_address or 'local'})")

    if args.resume:
        resume_path = str(Path(args.resume).resolve())
        logger.info(f"Loading previous results from {resume_path} to warm-start search")
        prior_grid = tune.ExperimentAnalysis(resume_path)
        points_to_evaluate, evaluated_rewards = [], []
        for trial in prior_grid.trials:
            if trial.last_result and trial.status == "TERMINATED":
                cfg = {k: trial.config[k] for k in hp_keys if k in trial.config}
                reward = trial.last_result.get(args.tune_metric)
                if cfg and reward is not None:
                    points_to_evaluate.append(cfg)
                    evaluated_rewards.append(reward)
        logger.info(f"Warm-starting from {len(points_to_evaluate)} prior trials, running {args.n_trials} new trials")
        search_alg = OptunaSearch(
            metric=args.tune_metric, mode="max", seed=args.seed,
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
        )

    TrialCounter.options(name="trial_counter", lifetime="detached", get_if_exists=True).remote()

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
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
    best_metric   = best.metrics[args.tune_metric]
    best_val_acc   = best.metrics.get("val_acc",   float("nan"))
    best_val_auroc = best.metrics.get("val_auroc", float("nan"))
    best_params    = {k: best.config[k] for k in hp_keys if k in best.config}

    all_trials = []
    completed = errored = 0
    for r in results:
        if r.error:
            errored += 1
            state = "ERROR"
        else:
            completed += 1
            state = "COMPLETE"
        all_trials.append({
            "val_acc":       r.metrics.get("val_acc")   if r.metrics else None,
            "val_auroc":     r.metrics.get("val_auroc") if r.metrics else None,
            args.tune_metric: r.metrics.get(args.tune_metric) if r.metrics else None,
            "params":        {k: r.config[k] for k in hp_keys if k in r.config} if r.config else {},
            "state":         state,
        })

    # Final-model phase: train on all data with best HPs; optionally save checkpoint
    # and/or evaluate on test data.
    test_results = {}
    if args.no_final:
        logger.info("Skipping final model training (--no-final).")
    elif args.checkpoint or args.test_data:
        test_data_path = Path(args.test_data) if args.test_data else None
        if test_data_path:
            logger.info(f"Training final model and evaluating on test data: {test_data_path}")
        else:
            logger.info("Training final model on all data (--checkpoint).")
        test_results = _evaluate_on_test_data(data_path, test_data_path, best_params, args)
        if test_data_path and test_results:
            logger.info(f"Test results: acc={test_results['test_acc']:.4f}, auroc={test_results['test_auroc']:.4f}")
    else:
        logger.info("Skipping final model training (no --checkpoint and no --test-data).")

    output = {
        "best_val_acc":        best_val_acc,
        "best_val_auroc":      best_val_auroc,
        f"best_{args.tune_metric}": best_metric,
        "best_params":         best_params,
        "model":               args.model,
        "dataset":             data_root,
        "num_classes":         num_classes,
        "n_trials":            args.n_trials,
        "epochs_max":          args.epochs,
        "n_folds":             args.folds,
        "n_repeats":           args.repeats,
        "stratified":          args.stratified,
        "pooling":             args.pooling,
        "tune_metric":         args.tune_metric,
        "completed_trials":    completed,
        "errored_trials":      errored,
        "total_time_seconds":  total_time,
        "all_trials":          all_trials,
    }
    if test_results:
        output["test_acc"] = test_results["test_acc"]
        output["test_auroc"] = test_results["test_auroc"]

    with open(f"{args.prefix}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest {args.tune_metric}: {best_metric:.4f}")
    if not np.isnan(best_val_acc):
        print(f"Best val_acc:   {best_val_acc:.4f}")
    if not np.isnan(best_val_auroc):
        print(f"Best val_auroc: {best_val_auroc:.4f}")
    if test_results:
        print(f"\nTest results:")
        print(f"  test_acc:   {test_results['test_acc']:.4f}")
        print(f"  test_auroc: {test_results['test_auroc']:.4f}")
    print("\nBest params:")
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
        "  epochs                 randint   [10, --epochs]            (epochs_min/epochs_max)\n"
        "  lr                     loguniform[1e-5, 1e-1]              (lr_min/lr_max)\n"
        "  weight_decay           loguniform[1e-6, 1e-2]              (wd_min/wd_max)\n"
        "  label_smoothing        uniform   [0.0, 0.2]                (ls_min/ls_max)\n"
        "  drop_rate              uniform   [0.0, 0.5]                (dr_min/dr_max)\n"
        "  randaugment_magnitude  randint   [0, 15]                   (ra_mag_min/ra_mag_max)\n"
        "  randaugment_num_ops    randint   [1, 3]                    (ra_ops_min/ra_ops_max)\n"
        "  mixup_alpha            uniform   [0.0, 0.4]                (mixup_min/mixup_max)\n"
        "  cutmix_alpha           uniform   [0.0, 1.0]                (cutmix_min/cutmix_max)\n"
        "  optimizer              choice    [AdamW, SGD]              (optimizers)\n"
    )

    p = argparse.ArgumentParser(
        description=f"cvic {_ver} — cross-validation hyperparameter search for image classifiers",
        formatter_class=_Fmt,
        epilog=epilog,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_ver}")
    p.add_argument("--data",    type=str, required=True,
                   help="Path to dataset root (ImageFolder layout with train/ subdirectory)")
    p.add_argument("--model",   type=str, default=None,
                   help="Any timm model name (e.g. resnet18, vit_small_patch16_224)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Use timm pretrained weights")
    p.add_argument("--n-trials",  type=int, default=30, dest="n_trials",
                   help="Number of Optuna trials")
    p.add_argument("--epochs",    type=int, default=30,
                   help="Upper bound for tunable epochs (default range: 10..--epochs); override via --search-space (epochs_min/epochs_max)")
    p.add_argument("--folds",     type=int, default=5,
                   help="Number of CV folds")
    p.add_argument("--repeats",   type=int, default=1,
                   help="Number of times to repeat the full CV with different random splits")
    p.add_argument("--stratified", action=argparse.BooleanOptionalAction, default=True,
                   help="Use stratified (class-balanced) folds")
    p.add_argument("--pooling",   action="store_true", default=False,
                   help="Pool out-of-fold predictions and compute metric once (default: average per-fold metrics)")
    p.add_argument("--tune-metric", type=str, default="val_auroc", dest="tune_metric",
                   help="Metric for Optuna trial selection")
    p.add_argument("--batch-size",  type=int, default=32, dest="batch_size",
                   help="Batch size per fold")
    p.add_argument("--prefix",    type=str, default="cvic",
                   help="Prefix for output files")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed")
    p.add_argument("--device",    type=str, default="auto",
                   help="Device: auto, cuda, mps, or cpu")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU-only mode (useful for reproducible results)")
    p.add_argument("--repro", action="store_true",
                   help="Enable reproducibility mode (deterministic algorithms, no cuDNN benchmark); trades performance for exact reproducibility")
    p.add_argument("--workers",   type=int, default=4,
                   help="DataLoader worker count")
    p.add_argument("--img-size",  type=int, default=224, dest="img_size",
                   help="Input image resolution")
    p.add_argument("--freeze-backbone", type=int, default=0, dest="freeze_backbone",
                   help="Epochs to freeze backbone; 0 = no freeze")
    p.add_argument("--amp",       action="store_true", default=False,
                   help="Enable automatic mixed precision (BF16 on H100/A100, FP16 on older GPUs)")
    p.add_argument("--search-space", type=str, default=None, dest="search_space",
                   help="YAML file to override search space bounds")
    p.add_argument("--ray-address", type=str, default=None, dest="ray_address",
                   help="Ray cluster address (default: start local cluster)")
    p.add_argument("--ray-storage", type=str, default=None, dest="ray_storage",
                   help="Ray Tune storage path for trial checkpoints")
    p.add_argument("--test-data", type=str, default=None, dest="test_data",
                   help="Optional test set path (ImageFolder layout); if given, trains final model on all data using best hyperparams")
    p.add_argument("--no-final", action="store_true", dest="no_final",
                   help="Skip final-model training after HPO (overrides --checkpoint and --test-data triggers); HPO-only run")
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="Save the final-model checkpoint (.pt) after training on all data with best HPs. "
                        "Triggers final-model training even without --test-data.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a previous Ray Tune experiment directory; warm-starts Optuna search from those results and runs --n-trials new trials")
    p.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                   help="Run end-to-end smoke test with synthetic data")
    p.add_argument("--random-seed", action="store_true", dest="random_seed",
                   help="Use a random seed derived from wallclock time (ignores --seed)")
    return p.parse_args()


def run_smoke_test(args):
    """End-to-end smoke test on synthetic data — quick sanity check that nothing is broken."""
    import numpy as np
    import tempfile
    from PIL import Image

    logger.info("Running smoke test...")
    _ck().set_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_dir = tmpdir / "train"
        for cls in ["cat", "dog", "bird"]:
            cls_dir = train_dir / cls
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
            folds=3,
            repeats=1,
            stratified=True,
            pooling=False,
            tune_metric="val_auroc",
            batch_size=4,
            prefix="smoke",
            seed=0,
            device=args.device,
            cpu=False,
            repro=False,
            workers=0,
            img_size=64,
            freeze_backbone=0,
            amp=False,
            search_space=None,
            ray_address=None,
            ray_storage=None,
            test_data=None,
            no_final=True,
            checkpoint=False,
            resume=None,
            smoke_test=False,
            random_seed=False,
        )
        run_cv(smoke_args)
        logger.info("Smoke test completed successfully")


def main():
    _t_program_start = time.time()
    try:
        args = parse_args()
        if args.random_seed:
            args.seed = int(time.time() * 1e6) % (2**31)
            logger.info(f"Generated random seed: {args.seed}")
        if args.smoke_test:
            run_smoke_test(args)
            return
        run_cv(args)
    finally:
        print(f"Total time: {_ck().format_duration(time.time() - _t_program_start)}")


if __name__ == "__main__":
    main()
