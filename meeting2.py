"""Generate, compare, and test text-to-image pools for course Meeting 2."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
POOLS_DIR = ROOT / "pools"
RESULTS_DIR = ROOT / "results" / "meeting2"
SPLITS_DIR = ROOT / "manifests" / "meeting2"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
IMAGE_COUNT = 110
SYNTHETIC_BUDGET = 55
IMAGE_SIZE = 512
SEEDS = (1729, 2718, 31415, 16180, 57721)
BASE_GUIDANCE = 7.5
BASE_STEPS = 30
POOL_SETTINGS = {
    "m2-base": {},
    "m2-guidance-2": {"guidance": 2.0},
    "m2-guidance-15": {"guidance": 15.0},
    "m2-steps-10": {"steps": 10},
}
NEGATIVE_PROMPT = (
    "illustration, painting, cartoon, text, watermark, multiple cats, "
    "deformed animal, extra limbs"
)
METRIC_NAMES = (
    "mau_recall",
    "macro_f1",
    "accuracy",
    "mau_precision",
    "mau_f1",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pool_path_from_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def settings_for_pool(name: str, prompts: list[str], args: argparse.Namespace) -> dict:
    variation = POOL_SETTINGS[name]
    return {
        "name": name,
        "mode": "t2i",
        "model": "sd15",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompts": prompts,
        "negative_prompt": args.negative_prompt,
        "steps": variation.get("steps", BASE_STEPS),
        "guidance": variation.get("guidance", BASE_GUIDANCE),
        "scheduler": "dpm",
        "seed": args.seed,
        "count": args.count,
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
    }


def select_device(torch, requested: str, allow_cpu: bool):
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cpu":
        if not allow_cpu:
            raise RuntimeError(
                "No GPU is available. CPU generation is impractically slow; "
                "use a GPU environment or pass --allow-cpu to opt in explicitly."
            )
        print(
            "Warning: Stable Diffusion generation on CPU is extremely slow. "
            "A CUDA GPU or a hosted GPU notebook is strongly recommended.",
            file=sys.stderr,
            flush=True,
        )
    return device


def load_text2image_pipeline(torch, settings: dict, device, low_vram: bool):
    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
    )
    from diffusers.utils import logging as diffusers_logging
    from transformers.utils import logging as transformers_logging

    diffusers_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    load_options = {
        "revision": settings["model_revision"],
        "dtype": dtype,
    }
    if dtype == torch.float16:
        load_options["variant"] = "fp16"
    pipeline = AutoPipelineForText2Image.from_pretrained(
        settings["model_id"], **load_options
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    if low_vram and device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    if low_vram:
        pipeline.enable_attention_slicing()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def draw_contact_sheet(pool_dir: Path, records: list[dict]) -> None:
    cell_w, cell_h = 256, 286
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 6), "white")
    draw = ImageDraw.Draw(sheet)
    for position, record in enumerate(records[:24]):
        x, y = (position % 4) * cell_w, (position // 4) * cell_h
        file_name = record.get("file")
        if file_name and (pool_dir / file_name).is_file():
            with Image.open(pool_dir / file_name) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (cell_w, cell_h - 30))
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        else:
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 31), fill="#dddddd")
        status = record.get("status", "missing")
        draw.text(
            (x + 5, y + cell_h - 26),
            f"{record['index']:03d} seed={record['seed']} {status}"[:38],
            fill="black",
        )
    sheet.save(pool_dir / "contact-sheet.jpg", quality=90)


def generate(args: argparse.Namespace) -> None:
    import diffusers
    import torch

    device = select_device(torch, args.device, args.allow_cpu)
    names = list(POOL_SETTINGS) if "all" in args.pools else args.pools
    prompts = [prompt.strip() for prompt in args.prompt]
    if any(not prompt for prompt in prompts):
        raise ValueError("Prompts must not be empty.")
    settings = {
        name: settings_for_pool(name, prompts, args)
        for name in names
    }
    generator_environment = {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(
            torch.float16 if device.type in ("cuda", "mps") else torch.float32
        ).removeprefix("torch."),
        "torch": str(torch.__version__),
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "low_vram": args.low_vram,
    }
    for settings_item in settings.values():
        settings_item.update(generator_environment)
    for name in names:
        settings_path = POOLS_DIR / name / "pool.json"
        if settings_path.is_file():
            saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if saved_settings != settings[name]:
                raise ValueError(
                    f"{POOLS_DIR / name} already contains a different configuration. "
                    "Choose a new pool name or keep the original settings."
                )
    pipeline = load_text2image_pipeline(
        torch, settings[names[0]], device, args.low_vram
    )
    default_scheduler_config = pipeline.scheduler.config

    for name in names:
        pool_dir = POOLS_DIR / name
        pool_dir.mkdir(parents=True, exist_ok=True)
        settings_path = pool_dir / "pool.json"
        if not settings_path.exists():
            write_json(settings_path, settings[name])

        manifest_path = pool_dir / "manifest.jsonl"
        records = read_jsonl(manifest_path)
        completed = {int(record["index"]) for record in records}
        pipeline.scheduler = type(pipeline.scheduler).from_config(
            default_scheduler_config
        )
        print(
            f"{name}: {len(completed)}/{args.count} attempts already recorded "
            f"on {device}",
            flush=True,
        )
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for index in range(args.count):
                if index in completed:
                    continue
                seed = args.seed + index
                prompt = prompts[index % len(prompts)]
                image_path = pool_dir / "images" / f"{name}-{index:04d}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = image_path.with_suffix(".tmp.png")
                attempt = {
                    "index": index,
                    "sample_id": f"{name}-{index:04d}",
                    **settings[name],
                    "prompt": prompt,
                    "seed": seed,
                    "scheduler_class": type(pipeline.scheduler).__name__,
                }
                started = time.monotonic()
                try:
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt or None,
                        num_inference_steps=settings[name]["steps"],
                        guidance_scale=settings[name]["guidance"],
                        width=IMAGE_SIZE,
                        height=IMAGE_SIZE,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    )
                    image = result.images[0]
                    flags = getattr(result, "nsfw_content_detected", None)
                    blocked = bool(flags and flags[0])
                    image.save(temporary)
                    temporary.replace(image_path)
                    record = {
                        **attempt,
                        "file": f"images/{image_path.name}",
                        "status": "safety_blocked" if blocked else "ok",
                        "seconds": round(time.monotonic() - started, 2),
                        "sha256": sha256_file(image_path),
                    }
                except (OSError, RuntimeError) as error:
                    if image_path.exists():
                        image_path.unlink()
                    if temporary.exists():
                        temporary.unlink()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    record = {
                        **attempt,
                        "file": None,
                        "status": "error",
                        "seconds": round(time.monotonic() - started, 2),
                        "error": f"{type(error).__name__}: {error}",
                    }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                completed.add(index)
                print(
                    f"  {index + 1}/{args.count}: {record['status']} "
                    f"({record['seconds']}s)",
                    flush=True,
                )
        records = read_jsonl(manifest_path)
        draw_contact_sheet(
            pool_dir, sorted(records, key=lambda record: int(record["index"]))
        )
        usable = sum(record["status"] == "ok" for record in records)
        print(
            f"{name}: {usable} usable images; contact sheet: "
            f"{pool_dir / 'contact-sheet.jpg'}",
            flush=True,
        )
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()


def read_selection() -> list[dict]:
    path = ROOT / "data" / "selection.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected data manifest: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def prepare_splits() -> list[list[dict]]:
    rows = read_selection()
    if len(rows) != 910 or len({row["path"] for row in rows}) != 910:
        raise ValueError("Expected 910 uniquely named selected original images.")
    for row in rows:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Selected original image hash mismatch: {row['path']}")
    manifests = []
    for iteration, seed in enumerate(SEEDS, start=1):
        rng = np.random.default_rng(seed)
        split = []
        for label in range(8):
            pool = [row for row in rows if row["label"] == label]
            for offset, index in enumerate(rng.permutation(len(pool))):
                role = "test" if offset < 30 else "val" if offset < 50 else "train"
                split.append({**pool[int(index)], "split": role})
        if len(split) != 910:
            raise ValueError(f"Iteration {iteration} does not contain 910 images.")
        for label in range(8):
            counts = {
                role: sum(
                    row["label"] == label and row["split"] == role
                    for row in split
                )
                for role in ("train", "val", "test")
            }
            expected = {
                "train": 20 if label == 4 else 70,
                "val": 20,
                "test": 30,
            }
            if counts != expected:
                raise ValueError(
                    f"Unexpected split counts for label {label}: {counts}"
                )
        path = SPLITS_DIR / f"iteration_{iteration}.csv"
        write_csv(path, split)
        manifests.append(split)
    return manifests


def pool_records(name: str) -> list[dict]:
    pool_dir = POOLS_DIR / name
    settings_file = pool_dir / "pool.json"
    manifest_file = pool_dir / "manifest.jsonl"
    if not settings_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(
            f"Pool {name!r} is incomplete. Generate it first with "
            "`python meeting2.py generate`."
        )
    records = read_jsonl(manifest_file)
    usable = []
    for record in records:
        if record.get("status") != "ok" or not record.get("file"):
            continue
        path = pool_dir / record["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest marks {path} as usable but the image is missing."
            )
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Generated image hash mismatch: {path}")
        usable.append(record)
    if len(usable) < SYNTHETIC_BUDGET:
        raise ValueError(
            f"Pool {name!r} has only {len(usable)} usable images; "
            f"at least {SYNTHETIC_BUDGET} are required."
        )
    return usable


def load_features(pool_names: list[str], threads: int):
    original_rows = read_selection()
    base_cache = ROOT / "features.npz"
    if not base_cache.is_file():
        raise FileNotFoundError(
            "Missing features.npz; prepare the original data and feature cache "
            "before running Meeting 2."
        )
    with np.load(base_cache) as cache:
        base_paths = [str(path) for path in cache["paths"].tolist()]
        base_hashes = [str(value) for value in cache["sha256"].tolist()]
        base_features = cache["X"].copy()
    base_lookup = {path: i for i, path in enumerate(base_paths)}
    for row in original_rows:
        index = base_lookup.get(row["path"])
        if index is None or base_hashes[index] != row["sha256"]:
            raise ValueError(
                f"features.npz does not match the selected originals at {row['path']}."
            )
    original_features = {
        row["path"]: base_features[base_lookup[row["path"]]]
        for row in original_rows
    }

    image_paths = []
    image_hashes = []
    image_ids = {}
    for name in pool_names:
        records = pool_records(name)
        for record in records:
            path = pool_path_from_root(POOLS_DIR / name / record["file"])
            image_paths.append(path)
            image_hashes.append(record["sha256"])
            image_ids[(name, record["sample_id"])] = len(image_paths) - 1

    cache_path = ROOT / "meeting2_features.npz"
    cached_features = None
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            if (
                list(cache["paths"].tolist()) == image_paths
                and list(cache["sha256"].tolist()) == image_hashes
            ):
                cached_features = cache["X"].copy()
    if cached_features is None:
        from experiment import extract

        cached_features, environment = extract(image_paths, threads)
        np.savez_compressed(
            cache_path,
            paths=image_paths,
            sha256=image_hashes,
            X=cached_features,
        )
        import importlib.metadata as metadata

        environment.update(
            python=platform.python_version(),
            platform=platform.platform(),
            packages={
                package: metadata.version(package)
                for package in ("numpy", "Pillow", "scikit-learn", "matplotlib")
            },
        )
        write_json(RESULTS_DIR / "environment.json", environment)
    return original_rows, original_features, cached_features, image_ids


def score_predictions(y_true: list[int], probabilities: np.ndarray) -> tuple[dict, np.ndarray]:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=list(range(8)),
        target_names=[
            "Abyssinian", "Bengal", "Birman", "Bombay",
            "Egyptian_Mau", "Ragdoll", "Russian_Blue", "Siamese",
        ],
        output_dict=True,
        zero_division=0,
    )
    mau = report["Egyptian_Mau"]
    return (
        {
            "mau_recall": float(mau["recall"]),
            "macro_f1": float(f1_score(y_true, predictions, average="macro")),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "mau_precision": float(mau["precision"]),
            "mau_f1": float(mau["f1-score"]),
        },
        predictions,
    )


def predict(model, scaler, features: np.ndarray) -> np.ndarray:
    scores = scaler.transform(features) @ model.coef_.T + model.intercept_
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def save_predictions(path: Path, rows: list[dict], probabilities: np.ndarray) -> None:
    names = (
        "Abyssinian", "Bengal", "Birman", "Bombay", "Egyptian_Mau",
        "Ragdoll", "Russian_Blue", "Siamese",
    )
    predictions = probabilities.argmax(axis=1)
    write_csv(
        path,
        [
            {
                "path": row["path"],
                "true": row["label"],
                "predicted": int(prediction),
                **{
                    f"p_{name}": float(probability)
                    for name, probability in zip(names, distribution, strict=True)
                },
            }
            for row, prediction, distribution in zip(
                rows, predictions, probabilities, strict=True
            )
        ],
    )


def train(args: argparse.Namespace) -> None:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("error", category=ConvergenceWarning)
    names = list(POOL_SETTINGS) if "all" in args.pools else args.pools
    manifests = prepare_splits()
    original_rows, original_features, pool_features, image_ids = load_features(
        names, args.threads
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = ["baseline", *names]

    for condition in conditions:
        run_dir = RESULTS_DIR / condition
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_results = []
        for iteration, (seed, split) in enumerate(zip(SEEDS, manifests, strict=True), 1):
            train_rows = [row for row in split if row["split"] == "train"]
            validation_rows = [row for row in split if row["split"] == "val"]
            scaler = StandardScaler().fit(
                np.stack([original_features[row["path"]] for row in train_rows])
            )
            selected = []
            if condition != "baseline":
                available = pool_records(condition)
                rng = random.Random(seed + 100_000)
                selected = rng.sample(available, SYNTHETIC_BUDGET)
            train_features = [
                original_features[row["path"]] for row in train_rows
            ]
            train_labels = [row["label"] for row in train_rows]
            observations = [
                {"path": row["path"], "label": row["label"]}
                for row in train_rows
            ]
            if selected:
                train_features.extend(
                    pool_features[image_ids[(condition, record["sample_id"])]]
                    for record in selected
                )
                train_labels.extend([4] * len(selected))
                observations.extend(
                    {
                        "path": pool_path_from_root(
                            POOLS_DIR / condition / record["file"]
                        ),
                        "label": 4,
                    }
                    for record in selected
                )
            model = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=3000, tol=1e-7
            )
            model.fit(
                scaler.transform(np.stack(train_features)),
                train_labels,
            )
            folder = run_dir / f"seed-{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / "classifier.npz",
                coef=model.coef_,
                intercept=model.intercept_,
                classes=model.classes_,
                mean=scaler.mean_,
                scale=scaler.scale_,
            )
            write_csv(
                folder / "train_observations.csv",
                [
                    {"observation": index, **observation}
                    for index, observation in enumerate(observations)
                ],
            )
            validation_features = np.stack(
                [original_features[row["path"]] for row in validation_rows]
            )
            probabilities = predict(model, scaler, validation_features)
            metrics, _ = score_predictions(
                [row["label"] for row in validation_rows], probabilities
            )
            save_predictions(folder / "val_predictions.csv", validation_rows, probabilities)
            result = {
                "seed": seed,
                "iteration": iteration,
                "n_train": len(observations),
                "added_images": [record["sample_id"] for record in selected],
                **metrics,
            }
            write_json(folder / "val_metrics.json", result)
            seed_results.append(result)
            print(
                f"{condition} seed {seed}: validation Mau recall "
                f"{metrics['mau_recall']:.3f}, macro-F1 {metrics['macro_f1']:.3f}",
                flush=True,
            )
        result_file = {
            "name": condition,
            "method": "real" if condition == "baseline" else "synthetic",
            "pool": None if condition == "baseline" else condition,
            "classifier": {
                "features": "frozen ResNet-18 ImageNet1K_V1",
                "model": "scikit-learn LogisticRegression",
                "C": 1.0,
                "solver": "lbfgs",
                "tol": 1e-7,
                "max_iter": 3000,
            },
            "budget": 0 if condition == "baseline" else SYNTHETIC_BUDGET,
            "seeds": seed_results,
        }
        if condition != "baseline":
            result_file["pool_settings"] = json.loads(
                (POOLS_DIR / condition / "pool.json").read_text(encoding="utf-8")
            )
        write_json(run_dir / "results.json", result_file)
    make_report()


def mean_std(results: list[dict], metric: str) -> tuple[float, float]:
    values = np.array([entry[metric] for entry in results], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def make_report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_names = ["baseline", *POOL_SETTINGS]
    runs = {}
    for name in run_names:
        path = RESULTS_DIR / name / "results.json"
        if path.is_file():
            runs[name] = json.loads(path.read_text(encoding="utf-8"))
    if "baseline" not in runs:
        raise FileNotFoundError("Train the baseline before generating Meeting 2 reports.")

    metrics_rows = []
    for name, run in runs.items():
        for entry in run["seeds"]:
            metrics_rows.append(
                {"condition": name, **{key: entry[key] for key in ("seed", *METRIC_NAMES)}}
            )
    write_csv(RESULTS_DIR / "validation_metrics.csv", metrics_rows)
    baseline = {
        entry["seed"]: entry for entry in runs["baseline"]["seeds"]
    }
    paired_rows = []
    for name, run in runs.items():
        if name == "baseline":
            continue
        for entry in run["seeds"]:
            reference = baseline[entry["seed"]]
            paired_rows.append(
                {
                    "condition": name,
                    "seed": entry["seed"],
                    **{
                        f"delta_{metric}": entry[metric] - reference[metric]
                        for metric in METRIC_NAMES
                    },
                }
            )
    if paired_rows:
        write_csv(RESULTS_DIR / "validation_paired_differences.csv", paired_rows)

    labels = {"baseline": "Baseline", **{name: name.removeprefix("m2-") for name in POOL_SETTINGS}}
    text = [
        "# Meeting 2 — text-to-image generation",
        "",
        "## Protocol",
        "",
        "All pools use Stable Diffusion 1.5, DPM-Solver, the same prompt schedule, "
        "negative prompt, and per-image seeds. Each comparison changes one setting "
        "from the base: guidance 2, guidance 15, or 10 denoising steps. Every "
        "attempt is retained in its pool manifest; only successful, unblocked "
        "images are eligible for training.",
        "",
        "The existing pets_project classifier is retained: frozen ImageNet ResNet-18 "
        "features and logistic regression. Each pool contributes 55 images, selected "
        "without replacement per split. Five paired holdout seeds are evaluated; "
        "the first three use the Meeting 1 seeds, and two additional seeds expand "
        "validation evidence. This is not the template's EfficientNet fine-tuning "
        "procedure, so compare the paired runs within this repository.",
        "",
        "## Validation results",
        "",
        "Mean ± sample standard deviation over five paired splits. Validation is used "
        "to choose a pool; test results are not used for selection.",
        "",
        "| Condition | Mau recall | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, run in runs.items():
        values = run["seeds"]
        means = []
        for metric in METRIC_NAMES[:3]:
            mean, std = mean_std(values, metric)
            means.append(f"{mean * 100:.2f}% ± {std * 100:.2f}%")
        text.append(f"| {labels[name]} | " + " | ".join(means) + " |")
    text.extend(
        [
            "",
            "## Generation attempts",
            "",
            "| Pool | Usable | Safety blocked | Errors | Example generation error |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name in POOL_SETTINGS:
        records = read_jsonl(POOLS_DIR / name / "manifest.jsonl")
        if not records:
            continue
        errors = [
            record.get("error", "")
            for record in records
            if record.get("status") == "error"
        ]
        example = errors[0].replace("|", "/").replace("\n", " ") if errors else "—"
        text.append(
            f"| {name} | "
            f"{sum(record.get('status') == 'ok' for record in records)} | "
            f"{sum(record.get('status') == 'safety_blocked' for record in records)} | "
            f"{len(errors)} | {example} |"
        )
    text.extend(
        [
            "",
            "## Per-split paired changes from baseline",
            "",
            "Positive values favor the synthetic pool. Deltas are in percentage points.",
            "",
            "| Pool | Seed | Δ Mau recall | Δ Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for entry in paired_rows:
        text.append(
            f"| {labels[entry['condition']]} | {entry['seed']} | "
            f"{entry['delta_mau_recall'] * 100:+.2f} | "
            f"{entry['delta_macro_f1'] * 100:+.2f} |"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, title in zip(
        axes,
        ("mau_recall", "macro_f1"),
        ("Egyptian Mau recall", "Macro-F1 (8 classes)"),
        strict=True,
    ):
        for name, run in runs.items():
            entries = run["seeds"]
            axis.plot(
                [entry["seed"] for entry in entries],
                [entry[metric] * 100 for entry in entries],
                marker="o",
                label=labels[name],
            )
        axis.set(title=title, xlabel="Holdout seed", ylabel="%", ylim=(0, 100))
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_comparison.png", dpi=160)
    plt.close(fig)

    for name in POOL_SETTINGS:
        contact_sheet = POOLS_DIR / name / "contact-sheet.jpg"
        if contact_sheet.is_file():
            contact_sheet_link = Path(
                os.path.relpath(contact_sheet, RESULTS_DIR)
            ).as_posix()
            text.extend(
                [
                    "",
                    f"## Visual review: {name}",
                    "",
                    f"![Contact sheet: {name}]({contact_sheet_link})",
                    "",
                    "Review the sheet for wrong breed or coat, implausible anatomy, "
                    "illustration-like appearance, repeated poses/backgrounds, and "
                    "any blocked or failed generations before selecting this pool.",
                ]
            )
    test_path = RESULTS_DIR / "test-results.json"
    if test_path.is_file():
        tested = json.loads(test_path.read_text(encoding="utf-8"))
        decision = json.loads(
            (RESULTS_DIR / "decision.json").read_text(encoding="utf-8")
        )
        text.extend(
            [
                "",
                "## Selected pool and test results",
                "",
                f"Selected `{tested['pool']}` based on validation and contact-sheet "
                f"review. Rationale: {decision['reason']}",
                "",
                "| Condition | Mau recall | Macro-F1 | Accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, entries in (
            ("Real-only baseline (Meeting 1 protocol)", tested["baseline"]),
            (f"Meeting 2 {tested['pool']}", tested["seeds"]),
        ):
            means = [
                f"{mean_std(entries, metric)[0] * 100:.2f}% ± "
                f"{mean_std(entries, metric)[1] * 100:.2f}%"
                for metric in METRIC_NAMES[:3]
            ]
            text.append(f"| {label} | " + " | ".join(means) + " |")
        text.extend(
            [
                "",
                "![Selected-pool test comparison](test_comparison.png)",
                "",
                "Test results are reported only for the pool chosen from validation "
                "results and contact sheets.",
            ]
        )
        baseline_test = {entry["seed"]: entry for entry in tested["baseline"]}
        test_deltas = [
            {
                "condition": tested["pool"],
                "seed": entry["seed"],
                **{
                    f"delta_{metric}": entry[metric] - baseline_test[entry["seed"]][metric]
                    for metric in METRIC_NAMES
                },
            }
            for entry in tested["seeds"]
        ]
        write_csv(RESULTS_DIR / "test_paired_differences.csv", test_deltas)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, metric, title in zip(
            axes,
            ("mau_recall", "macro_f1"),
            ("Egyptian Mau recall", "Macro-F1 (8 classes)"),
            strict=True,
        ):
            for label, entries in (
                ("Real-only baseline", tested["baseline"]),
                (f"Selected: {tested['pool']}", tested["seeds"]),
            ):
                axis.plot(
                    [entry["seed"] for entry in entries],
                    [entry[metric] * 100 for entry in entries],
                    marker="o",
                    label=label,
                )
            axis.set(title=title, xlabel="Holdout seed", ylabel="%", ylim=(0, 100))
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "test_comparison.png", dpi=160)
        plt.close(fig)

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Meeting 2 report written to {report_path}")


def test_pool(args: argparse.Namespace) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    result_path = RESULTS_DIR / args.pool / "results.json"
    baseline_path = RESULTS_DIR / "baseline" / "results.json"
    if not result_path.is_file() or not baseline_path.is_file():
        raise FileNotFoundError(
            f"Train pool {args.pool!r} and baseline before testing."
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_result = json.loads(baseline_path.read_text(encoding="utf-8"))
    for run in (result, baseline_result):
        if [entry["seed"] for entry in run["seeds"]] != list(SEEDS):
            raise ValueError(
                f"{run['name']} is missing one or more Meeting 2 validation runs."
            )
    if len(args.reason.strip()) < 10:
        raise ValueError(
            "Provide a short justification based on validation results and the "
            "contact sheet (at least 10 characters)."
        )
    manifests = prepare_splits()
    _, original_features, _, _ = load_features([args.pool], args.threads)
    test_results = []
    baseline_test_results = []

    for seed, split in zip(SEEDS, manifests, strict=True):
        rows = [row for row in split if row["split"] == "test"]
        folder = RESULTS_DIR / args.pool / f"seed-{seed}"
        with np.load(folder / "classifier.npz") as saved:
            coef = saved["coef"]
            intercept = saved["intercept"]
            mean = saved["mean"]
            scale = saved["scale"]
        features = np.stack([original_features[row["path"]] for row in rows])
        scores = ((features - mean) / scale) @ coef.T + intercept
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        metrics, predictions = score_predictions(
            [row["label"] for row in rows], probabilities
        )
        save_predictions(folder / "test_predictions.csv", rows, probabilities)
        matrix = confusion_matrix(
            [row["label"] for row in rows], predictions, labels=list(range(8))
        )
        write_csv(
            folder / "test_confusion.csv",
            [
                {
                    "actual": label,
                    **{
                        name: int(value)
                        for name, value in zip(
                            (
                                "Abyssinian", "Bengal", "Birman", "Bombay",
                                "Egyptian_Mau", "Ragdoll", "Russian_Blue", "Siamese",
                            ),
                            row,
                            strict=True,
                        )
                    },
                }
                for label, row in zip(
                    (
                        "Abyssinian", "Bengal", "Birman", "Bombay",
                        "Egyptian_Mau", "Ragdoll", "Russian_Blue", "Siamese",
                    ),
                    matrix,
                    strict=True,
                )
            ],
        )
        report = classification_report(
            [row["label"] for row in rows],
            predictions,
            labels=list(range(8)),
            output_dict=True,
            zero_division=0,
        )
        (folder / "test_classification_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        test_results.append({"seed": seed, **metrics})
        baseline_folder = RESULTS_DIR / "baseline" / f"seed-{seed}"
        with np.load(baseline_folder / "classifier.npz") as saved:
            baseline_coef = saved["coef"]
            baseline_intercept = saved["intercept"]
            baseline_mean = saved["mean"]
            baseline_scale = saved["scale"]
        baseline_scores = (
            (features - baseline_mean) / baseline_scale
        ) @ baseline_coef.T + baseline_intercept
        baseline_scores -= baseline_scores.max(axis=1, keepdims=True)
        baseline_probabilities = np.exp(baseline_scores)
        baseline_probabilities /= baseline_probabilities.sum(axis=1, keepdims=True)
        baseline_metrics, _ = score_predictions(
            [row["label"] for row in rows], baseline_probabilities
        )
        save_predictions(
            baseline_folder / "test_predictions.csv", rows, baseline_probabilities
        )
        baseline_test_results.append({"seed": seed, **baseline_metrics})
        print(
            f"{args.pool} seed {seed}: test Mau recall "
            f"{metrics['mau_recall']:.3f}, macro-F1 {metrics['macro_f1']:.3f}",
            flush=True,
        )
    test_payload = {
        "pool": args.pool,
        "reason": args.reason.strip(),
        "seeds": test_results,
        "baseline": [
            {
                "seed": entry["seed"],
                **{metric: entry[metric] for metric in METRIC_NAMES},
            }
            for entry in baseline_test_results
        ],
    }
    write_json(RESULTS_DIR / "test-results.json", test_payload)
    write_json(
        RESULTS_DIR / "decision.json",
        {
            "pool": args.pool,
            "selection_basis": "validation metrics and contact-sheet review",
            "reason": args.reason.strip(),
        },
    )
    make_report()


def main() -> int:
    global POOLS_DIR

    parser = argparse.ArgumentParser(
        description="Meeting 2 workflow: generate pools, train on validation, test a chosen pool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate", help="generate 110-image base and setting-variation pools"
    )
    generate_parser.add_argument(
        "--pools-dir",
        type=Path,
        default=Path("pools"),
        help="pool directory, relative to the project root unless absolute",
    )
    generate_parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="prompt; repeat to cycle through prompts in every pool",
    )
    generate_parser.add_argument(
        "--negative-prompt", default=NEGATIVE_PROMPT
    )
    generate_parser.add_argument(
        "--pools", nargs="+", choices=["all", *POOL_SETTINGS], default=["all"]
    )
    generate_parser.add_argument("--count", type=int, default=IMAGE_COUNT)
    generate_parser.add_argument("--seed", type=int, default=1000)
    generate_parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    generate_parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable memory-saving options on CUDA (default; use --no-low-vram to disable)",
    )
    generate_parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="explicitly opt in to impractically slow CPU generation",
    )
    train_parser = subparsers.add_parser(
        "train", help="train validation runs for baseline and generated pools"
    )
    train_parser.add_argument(
        "--pools-dir",
        type=Path,
        default=Path("pools"),
        help="pool directory, relative to the project root unless absolute",
    )
    train_parser.add_argument(
        "--pools", nargs="+", choices=["all", *POOL_SETTINGS], default=["all"]
    )
    train_parser.add_argument("--threads", type=int, default=4)
    test_parser = subparsers.add_parser(
        "test", help="test one pool chosen from validation and contact sheets"
    )
    test_parser.add_argument(
        "--pools-dir",
        type=Path,
        default=Path("pools"),
        help="pool directory, relative to the project root unless absolute",
    )
    test_parser.add_argument("--pool", required=True, choices=tuple(POOL_SETTINGS))
    test_parser.add_argument("--reason", required=True)
    test_parser.add_argument("--threads", type=int, default=4)
    report_parser = subparsers.add_parser(
        "report", help="regenerate the validation report and plots"
    )
    report_parser.add_argument(
        "--pools-dir",
        type=Path,
        default=Path("pools"),
        help="pool directory, relative to the project root unless absolute",
    )
    args = parser.parse_args()

    requested_pool_dir = (
        args.pools_dir if args.pools_dir.is_absolute() else ROOT / args.pools_dir
    )
    POOLS_DIR = Path(os.path.abspath(requested_pool_dir))
    try:
        POOLS_DIR.relative_to(ROOT)
    except ValueError:
        parser.error("--pools-dir must be inside the project directory.")

    if args.command == "generate":
        if args.count != IMAGE_COUNT:
            parser.error(f"Meeting 2 pools must contain exactly {IMAGE_COUNT} attempts.")
        generate(args)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test_pool(args)
    else:
        make_report()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Meeting 2 failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
