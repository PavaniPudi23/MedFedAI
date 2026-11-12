"""MedFedAI local model dashboard API."""

from __future__ import annotations

import base64
import csv
import importlib
import importlib.util
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
configured_data_root = Path(os.environ.get("MEDFEDAI_DATA_DIR", "data"))
DATA_ROOT = configured_data_root if configured_data_root.is_absolute() else ROOT / configured_data_root
app = FastAPI(title="MedFedAI Demo API", version="0.1.0")
RUNS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()
MODEL_RUNS: dict[str, dict[str, Any]] = {}
MODEL_RUN_LOCK = threading.Lock()
ACTIVE_MODEL_RUN: str | None = None
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CLIENTS = [
    {"id": "HOSP-01", "name": "Northside Hospital", "type": "Hospital", "patients": 1240, "color": "mint"},
    {"id": "CLIN-02", "name": "Riverbend Clinic", "type": "Clinic", "patients": 860, "color": "blue"},
    {"id": "CENT-03", "name": "Central Imaging Center", "type": "Imaging center", "patients": 1530, "color": "gold"},
]


class DemoConfig(BaseModel):
    rounds: int = Field(default=8, ge=3, le=20)
    sampling_rate: float = Field(default=0.67, gt=0, le=1)
    clip_norm: float = Field(default=1.0, gt=0, le=10)
    noise_multiplier: float = Field(default=1.1, ge=0, le=3)
    local_epochs: int = Field(default=1, ge=1, le=5)


class ModelConfig(BaseModel):
    dataset_id: str = "data"
    rounds: int = Field(default=20, ge=1, le=20)
    local_epochs: int = Field(default=10, ge=1, le=10)
    noise_multiplier: float = Field(default=1.0, ge=0, le=2)
    clip_norm: float = Field(default=1.0, gt=0, le=10)
    delta: float = Field(default=1e-5, gt=0, lt=1)


DATASET_LABELS = {
    "data": "Knee X-ray dataset",
    "data_chest": "Chest X-ray dataset",
}


def available_datasets() -> dict[str, Path]:
    """Expose only datasets the user imported through the local dashboard."""
    found: dict[str, Path] = {}
    if ROOT.is_dir():
        for path in ROOT.iterdir():
            if path.is_dir() and path.name.startswith("data_manual_") and (path / "clients").is_dir() and (path / "test").is_dir():
                found[path.name] = path
    return dict(sorted(found.items()))


def get_dataset_root(dataset_id: str) -> Path:
    dataset_root = available_datasets().get(dataset_id)
    if dataset_root is None:
        raise HTTPException(status_code=400, detail="Choose one of the prepared datasets shown in the dashboard.")
    return dataset_root


def dataset_output_root(dataset_id: str) -> Path:
    return ROOT / "outputs" / dataset_id


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-model-training"}


@app.get("/api/overview")
async def overview(dataset_id: str | None = Query(default=None)) -> dict[str, Any]:
    datasets = available_datasets()
    selected_id = dataset_id if dataset_id in datasets else (DATA_ROOT.name if DATA_ROOT.name in datasets else next(iter(datasets), ""))
    selected_root = datasets.get(selected_id)
    if selected_root is None:
        return {"project": "MedFedAI", "datasets": [], "dataset_id": "", "dataset": "No prepared dataset", "model_ready": False, "test_image_count": 0, "checkpoint_ready": False, "clients": [], "notice": "Prepare a dataset with client and test folders first."}
    client_root = selected_root / "clients"
    client_dirs = sorted(path for path in client_root.glob("client_*") if path.is_dir())
    clients = []
    for index, path in enumerate(client_dirs):
        image_count = sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES)
        clients.append({
            "id": path.name,
            "name": f"Simulated site {path.name}",
            "type": "Local client",
            "image_count": image_count,
            "color": ("mint", "blue", "gold")[index % 3],
        })
    test_root = selected_root / "test"
    test_images = [file for file in test_root.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES] if test_root.is_dir() else []
    model_ready = len(client_dirs) >= 2 and bool(test_images)
    checkpoints = list(dataset_output_root(selected_id).glob("**/deep_capsnet_fedavg_dp.pt"))
    if selected_id == "data":
        checkpoints.extend(path for path in (ROOT / "outputs").glob("*/deep_capsnet_fedavg_dp.pt") if path.is_file())
    return {
        "project": "MedFedAI",
        "mode": "local model training",
        "dataset_id": selected_id,
        "dataset": dataset_label(selected_id, selected_root),
        "datasets": [
            {"id": key, "label": dataset_label(key, path)}
            for key, path in datasets.items()
        ],
        "model_ready": model_ready,
        "test_image_count": len(test_images),
        "checkpoint_ready": bool(checkpoints),
        "clients": clients,
        "model": {"name": "Deep Capsule Network", "features": ["Convolutional feature layers", "Primary capsules", "Dynamic routing", "Margin loss"]},
        "federated_learning": "Clients train locally; FedAvg combines model updates at the server.",
        "privacy": {"mechanism": "Central client-level differential privacy", "controls": ["Client update clipping", "Gaussian noise", "Opacus epsilon/delta accounting"]},
        "notice": "When run locally, the training process reads images from local client folders and does not upload image files. The server process sees raw client updates before aggregate noise is added.",
    }


def dataset_label(dataset_id: str, dataset_root: Path) -> str:
    metadata_file = dataset_root / "dataset.json"
    if metadata_file.is_file():
        try:
            import json
            value = json.loads(metadata_file.read_text(encoding="utf-8"))
            if value.get("name"):
                return str(value["name"])
        except (OSError, ValueError):
            pass
    return DATASET_LABELS.get(dataset_id, dataset_id.replace("_", " ").title())


@app.post("/api/datasets/import", status_code=201)
async def import_dataset(
    request: Request,
    dataset_name: str = Query(default="My medical image dataset", max_length=80),
    client_count: int = Query(default=3, ge=2, le=10),
) -> dict[str, Any]:
    """Import a user-selected image folder into local simulated-client partitions."""
    class BodyReader:
        def __init__(self, body_request: Request):
            self.iterator = body_request.stream().__aiter__()
            self.buffer = bytearray()

        async def fill(self) -> bool:
            try:
                self.buffer.extend(await self.iterator.__anext__())
                return True
            except StopAsyncIteration:
                return False

        async def read_line(self, limit: int) -> bytes:
            while b"\n" not in self.buffer:
                if len(self.buffer) > limit or not await self.fill():
                    raise ValueError("The upload manifest is missing or too large.")
            split_at = self.buffer.index(b"\n")
            line = bytes(self.buffer[:split_at])
            del self.buffer[:split_at + 1]
            return line

        async def copy_exactly(self, output, byte_count: int, total_counter: list[int], limit: int) -> None:
            remaining = byte_count
            while remaining:
                if not self.buffer and not await self.fill():
                    raise ValueError("The upload ended before all selected image bytes arrived.")
                amount = min(remaining, len(self.buffer))
                chunk = self.buffer[:amount]
                del self.buffer[:amount]
                total_counter[0] += amount
                if total_counter[0] > limit:
                    raise HTTPException(status_code=413, detail="The selected image folder exceeds the 5 GB local import limit.")
                if output is not None:
                    output.write(chunk)
                remaining -= amount

    reader = BodyReader(request)
    try:
        manifest = json.loads((await reader.read_line(10 * 1024 * 1024)).decode("utf-8"))
        items = manifest.get("files", [])
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail="Could not read the selected folder manifest.") from error
    if not items or len(items) > 20000:
        raise HTTPException(status_code=400, detail="Choose between 1 and 20,000 image files.")
    total_bytes_expected = sum(int(item.get("size", -1)) for item in items)
    if total_bytes_expected < 0 or total_bytes_expected > 5 * 1024 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The selected image folder exceeds the 5 GB local import limit.")

    dataset_id = f"data_manual_{uuid.uuid4().hex[:8]}"
    dataset_root = ROOT / dataset_id
    incoming_root = dataset_root / ".incoming"
    image_records: list[tuple[str, Path]] = []
    total_bytes = [0]
    max_upload_bytes = 5 * 1024 * 1024 * 1024
    split_names = {"train", "training", "test", "testing", "val", "valid", "validation"}

    try:
        seen_paths: set[str] = set()
        for item in items:
            raw_relative = str(item.get("path", ""))
            normalized = raw_relative.replace("\\", "/")
            parts = [part for part in PurePosixPath(normalized).parts if part not in ("", ".")]
            if not parts or any(part == ".." for part in parts) or PurePosixPath(normalized).is_absolute():
                raise HTTPException(status_code=400, detail="A selected file has an invalid folder path.")
            try:
                size = int(item.get("size", -1))
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail="An image file has an invalid size.") from error
            if size < 0:
                raise HTTPException(status_code=400, detail="An image file has an invalid size.")
            if not parts[0].lower() in split_names and len(parts) >= 3:
                parts = parts[1:]
            safe_relative = "/".join(parts)
            if safe_relative in seen_paths:
                raise HTTPException(status_code=400, detail="The selected folder contains duplicate relative file paths.")
            seen_paths.add(safe_relative)
            suffix = Path(parts[-1]).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                await reader.copy_exactly(None, size, total_bytes, max_upload_bytes)
                continue
            if len(parts) < 2:
                raise HTTPException(status_code=400, detail="Images must be inside class folders (for example train/NORMAL/image.jpg).")
            target = incoming_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                await reader.copy_exactly(output, size, total_bytes, max_upload_bytes)
            image_records.append(("/".join(parts), target))
        if not image_records:
            raise HTTPException(status_code=400, detail="No supported image files were selected.")

        has_split_layout = any(path.split("/", 1)[0].lower() in split_names for path, _ in image_records)
        if has_split_layout:
            train_records = [(p, f) for p, f in image_records if p.split("/", 1)[0].lower() in {"train", "training"}]
            test_records = [(p, f) for p, f in image_records if p.split("/", 1)[0].lower() in {"test", "testing"}]
            validation_records = [(p, f) for p, f in image_records if p.split("/", 1)[0].lower() in {"val", "valid", "validation"}]
            source_records = train_records
        else:
            source_records = image_records
            test_records = []
            validation_records = []

        def by_class(records: list[tuple[str, Path]], split_layout: bool) -> dict[str, list[tuple[str, Path]]]:
            groups: dict[str, list[tuple[str, Path]]] = {}
            for relative, file_path in records:
                parts = relative.split("/")
                class_name = parts[1] if split_layout else parts[0]
                if class_name.startswith("."):
                    continue
                groups.setdefault(class_name, []).append((relative, file_path))
            return groups

        train_groups = by_class(source_records, has_split_layout)
        if len(train_groups) < 2:
            raise HTTPException(status_code=400, detail="The chosen folder must contain at least two class folders with images.")
        class_names = sorted(train_groups)
        if test_records:
            test_groups = by_class(test_records, True)
            if sorted(test_groups) != class_names or any(not test_groups[name] for name in class_names):
                raise HTTPException(status_code=400, detail="Train and test splits must contain the same class folders, each with images.")
        else:
            test_groups = {name: [] for name in class_names}

        import random as _random
        rng = _random.Random(17)
        train_counts: dict[str, dict[str, int]] = {f"client_{i}": {} for i in range(client_count)}
        test_counts: dict[str, int] = {}
        validation_counts: dict[str, int] = {}

        def move_record(record: tuple[str, Path], destination: Path, class_name: str, index: int) -> None:
            destination_dir = destination / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            source_file = record[1]
            target_file = destination_dir / f"{index:06d}_{source_file.name}"
            shutil.move(str(source_file), str(target_file))

        for class_name in class_names:
            train_images = list(train_groups[class_name])
            rng.shuffle(train_images)
            if not test_records:
                if len(train_images) < client_count + 1:
                    raise HTTPException(status_code=400, detail=f"Class {class_name} needs at least {client_count + 1} images to create clients and a test split.")
                test_size = max(1, round(len(train_images) * 0.2))
                test_groups[class_name] = train_images[:test_size]
                train_images = train_images[test_size:]
            if len(train_images) < client_count:
                raise HTTPException(status_code=400, detail=f"Class {class_name} has too few training images for {client_count} clients.")
            for image_index, record in enumerate(test_groups[class_name]):
                move_record(record, dataset_root / "test", class_name, image_index)
                test_counts[class_name] = test_counts.get(class_name, 0) + 1
            for image_index, record in enumerate(train_images):
                client_index = image_index % client_count
                client_key = f"client_{client_index}"
                move_record(record, dataset_root / "clients" / client_key, class_name, image_index)
                train_counts[client_key][class_name] = train_counts[client_key].get(class_name, 0) + 1

        for class_name, records in by_class(validation_records, True).items():
            for image_index, record in enumerate(records):
                move_record(record, dataset_root / "validation", class_name, image_index)
                validation_counts[class_name] = validation_counts.get(class_name, 0) + 1

        shutil.rmtree(incoming_root, ignore_errors=True)
        clean_name = " ".join(dataset_name.split()).strip() or "My medical image dataset"
        metadata = {
            "name": clean_name,
            "dataset_id": dataset_id,
            "client_count": client_count,
            "source_layout": "provided train/test split" if any(p.split("/", 1)[0].lower() in {"test", "testing"} for p, _ in image_records) else "random image-level 80/20 train/test split",
            "patient_level_split": False,
            "client_class_counts": train_counts,
            "test_class_counts": test_counts,
            "validation_class_counts": validation_counts,
            "warning": "Patient-level separation is not verified. Client allocation is approximately IID.",
        }
        (dataset_root / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"dataset_id": dataset_id, "name": clean_name, "client_count": client_count, "test_image_count": sum(test_counts.values()), "source_layout": metadata["source_layout"]}
    except HTTPException:
        shutil.rmtree(dataset_root, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(dataset_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not import dataset: {error}") from error


def _image_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _activation_heatmap(values, size: int = 160):
    import numpy as np
    from PIL import Image, ImageOps

    array = values.detach().float().cpu().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = ((array - low) / (high - low) * 255).astype(np.uint8)
    grayscale = Image.fromarray(normalized).resize((size, size), Image.Resampling.BILINEAR)
    return ImageOps.colorize(grayscale, black="#123b3b", mid="#318d7c", white="#f2c777")


def _score_chart(class_names: list[str], scores: list[float]):
    from PIL import Image, ImageDraw, ImageFont

    width, row_height = 440, 35
    height = 42 + row_height * len(class_names)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_score = max(scores) if scores else 1.0
    for index, (name, score) in enumerate(zip(class_names, scores)):
        y = 18 + index * row_height
        draw.text((8, y + 3), name[:20], fill="#526568", font=font)
        bar_left, bar_right = 170, 385
        draw.rounded_rectangle((bar_left, y, bar_right, y + 20), radius=5, fill="#edf3f1")
        draw.rounded_rectangle((bar_left, y, bar_left + int((bar_right - bar_left) * max(0.0, min(1.0, score))), y + 20), radius=5, fill="#258a78")
        draw.text((bar_right + 8, y + 4), f"{score:.3f}", fill="#526568", font=font)
    return image


@app.get("/api/model/visualization")
async def model_visualization(index: int = Query(default=0, ge=0), dataset_id: str | None = Query(default=None)) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torchvision import datasets, transforms

    selected_id = dataset_id or DATA_ROOT.name
    test_root = get_dataset_root(selected_id) / "test"
    test_dataset = datasets.ImageFolder(test_root)
    if len(test_dataset) == 0:
        raise HTTPException(status_code=404, detail="No held-out test images were found.")
    checkpoint_paths = [path for path in dataset_output_root(selected_id).glob("**/deep_capsnet_fedavg_dp.pt") if path.is_file()]
    if selected_id == "data":
        checkpoint_paths.extend(path for path in (ROOT / "outputs").glob("*/deep_capsnet_fedavg_dp.pt") if path.is_file())
    if not checkpoint_paths:
        raise HTTPException(status_code=404, detail="Train the model first; no local checkpoint is available.")
    checkpoint_path = max(checkpoint_paths, key=lambda path: path.stat().st_mtime)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    if classes != test_dataset.classes:
        raise HTTPException(status_code=409, detail="The latest model checkpoint uses different class folders than the current test dataset. Train a matching model first.")

    model_directory = str(ROOT / "demo readme")
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)
    model_module = importlib.import_module("train_federated")
    model = model_module.DeepCapsNet(
        int(checkpoint["channels"]), len(classes), int(checkpoint.get("routing_iters", 3))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_index = index % len(test_dataset)
    source_path, label_index = test_dataset.samples[sample_index]
    image_size = int(checkpoint["image_size"])
    channels = int(checkpoint["channels"])
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1) if channels == 1 else transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ])
    with Image.open(source_path) as opened:
        original = opened.convert("L" if channels == 1 else "RGB")
        original.thumbnail((160, 160))
        model_image = image_transform(opened).unsqueeze(0)
    with torch.no_grad():
        class_scores, intermediate = model.forward_with_intermediates(model_image)
    scores = [float(value) for value in class_scores[0].tolist()]
    predicted_index = int(class_scores[0].argmax().item())
    feature_map = intermediate["feature_maps"][0].mean(dim=0)
    primary_map = intermediate["primary_capsules"][0].norm(dim=-1).reshape(4, 4, 8).mean(dim=-1)
    preprocessed = model_image[0, 0] if channels == 1 else model_image[0].mean(dim=0)
    preprocessed = ((preprocessed * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
    preprocessed_image = Image.fromarray(preprocessed).resize((160, 160), Image.Resampling.NEAREST)
    phases = [
        {"step": 1, "title": "Original image", "caption": "Held-out image selected from the local test folder.", "image": _image_data_uri(original.resize((160, 160)))},
        {"step": 2, "title": "Preprocessed image", "caption": f"Resized to {image_size} × {image_size} and normalized for the model.", "image": _image_data_uri(preprocessed_image)},
        {"step": 3, "title": "Convolution features", "caption": "Mean activation across the final convolution channels; brighter areas have higher activation.", "image": _image_data_uri(_activation_heatmap(feature_map))},
        {"step": 4, "title": "Primary capsule activity", "caption": "Capsule vector magnitudes arranged as a 4 × 4 activation map.", "image": _image_data_uri(_activation_heatmap(primary_map))},
        {"step": 5, "title": "Routing and class scores", "caption": "Capsule outputs after dynamic routing. Scores are not calibrated probabilities.", "image": _image_data_uri(_score_chart(classes, scores))},
    ]
    return {
        "sample_index": sample_index,
        "sample_count": len(test_dataset),
        "true_class": classes[label_index],
        "predicted_class": classes[predicted_index],
        "scores": dict(zip(classes, scores)),
        "checkpoint": checkpoint_path.parent.name,
        "phases": phases,
    }


def simulate_run(run_id: str, config: DemoConfig) -> None:
    """Create illustrative progress records; no model or dataset is accessed."""
    rng = random.Random(run_id)
    rounds = config.rounds
    with RUN_LOCK:
        run = RUNS[run_id]
        run["status"] = "running"
    for index in range(1, rounds + 1):
        time.sleep(0.55)
        progress = index / rounds
        accuracy = min(0.91, 0.55 + 0.31 * (1 - (0.78 ** index)) + rng.uniform(-0.018, 0.018))
        # More injected noise illustrates a utility trade-off in this mock view.
        accuracy -= max(0, config.noise_multiplier - 1.0) * 0.025
        loss = max(0.16, 0.92 * (0.79 ** index) + rng.uniform(-0.025, 0.025))
        sampled = [client for client in CLIENTS if rng.random() < config.sampling_rate]
        if not sampled:
            sampled = [rng.choice(CLIENTS)]
        record = {
            "round": index,
            "progress": round(progress * 100),
            "sampled_clients": [client["id"] for client in sampled],
            "client_count": len(sampled),
            "train_loss": round(loss, 4),
            "validation_accuracy": round(max(0, min(1, accuracy)), 4),
            "raw_images_transferred": 0,
            "protection": "clipping + Gaussian noise" if config.noise_multiplier > 0 else "privacy disabled",
        }
        with RUN_LOCK:
            run = RUNS[run_id]
            run["rounds_completed"] = index
            run["history"].append(record)
            run["latest"] = record
        # Brief delay gives the browser a visible round-by-round presentation.
    with RUN_LOCK:
        RUNS[run_id]["status"] = "completed"
        RUNS[run_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/demo/run", status_code=202)
async def start_demo(config: DemoConfig, background_tasks: BackgroundTasks) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    run = {
        "run_id": run_id,
        "status": "queued",
        "mode": "simulated review demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rounds_total": config.rounds,
        "rounds_completed": 0,
        "config": config.model_dump(),
        "history": [],
        "latest": None,
        "notice": "Simulated metrics for presentation only; no dataset or neural network training is performed.",
    }
    with RUN_LOCK:
        RUNS[run_id] = run
    background_tasks.add_task(simulate_run, run_id, config)
    return {"run_id": run_id, "status": "queued", "status_url": f"/api/demo/run/{run_id}"}


@app.get("/api/demo/run/{run_id}")
async def get_demo_run(run_id: str) -> dict[str, Any]:
    with RUN_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return dict(run)


def _run_model_process(run_id: str, config: ModelConfig, output_dir: Path, data_root: Path) -> None:
    global ACTIVE_MODEL_RUN
    script = ROOT / "demo readme" / "train_federated.py"
    command = [
        sys.executable, str(script),
        "--data-dir", str(data_root),
        "--output-dir", str(output_dir),
        "--rounds", str(config.rounds),
        "--local-epochs", str(config.local_epochs),
        "--batch-size", "32",
        "--image-size", "32",
        "--channels", "1",
        "--client-sampling-rate", "1.0",
        "--clip-norm", str(config.clip_norm),
        "--noise-multiplier", str(config.noise_multiplier),
        "--delta", str(config.delta),
    ]
    log_path = output_dir / "training.log"
    with MODEL_RUN_LOCK:
        MODEL_RUNS[run_id]["status"] = "running"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        with MODEL_RUN_LOCK:
            run = MODEL_RUNS[run_id]
            run["status"] = "completed" if completed.returncode == 0 else "failed"
            run["return_code"] = completed.returncode
            if completed.returncode != 0:
                run["error"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        with MODEL_RUN_LOCK:
            run = MODEL_RUNS[run_id]
            run["status"] = "failed"
            run["error"] = str(error)
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
    finally:
        with MODEL_RUN_LOCK:
            ACTIVE_MODEL_RUN = None


@app.post("/api/model/run", status_code=202)
async def start_model_run(config: ModelConfig, background_tasks: BackgroundTasks) -> dict[str, Any]:
    global ACTIVE_MODEL_RUN
    data_root = get_dataset_root(config.dataset_id)
    client_dirs = sorted(path for path in (data_root / "clients").glob("client_*") if path.is_dir())
    test_root = data_root / "test"
    script = ROOT / "demo readme" / "train_federated.py"
    if not script.is_file():
        raise HTTPException(status_code=500, detail="The model training script is missing.")
    if len(client_dirs) < 2 or not test_root.is_dir():
        raise HTTPException(status_code=400, detail=f"Prepare {config.dataset_id}/clients/client_*/<class>/ and {config.dataset_id}/test/<class>/ first.")
    if not any(file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES for folder in client_dirs for file in folder.rglob("*")):
        raise HTTPException(status_code=400, detail="No supported images were found in the client folders.")
    if not any(file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES for file in test_root.rglob("*")):
        raise HTTPException(status_code=400, detail=f"No supported images were found in {config.dataset_id}/test.")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("opacus") is None:
        raise HTTPException(status_code=503, detail="Install the model dependencies in the same Python environment as the web server. See MODEL_README.md.")

    with MODEL_RUN_LOCK:
        if ACTIVE_MODEL_RUN is not None:
            raise HTTPException(status_code=409, detail="A real model run is already in progress.")
        run_id = str(uuid.uuid4())
        output_dir = dataset_output_root(config.dataset_id) / f"web_{run_id[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        ACTIVE_MODEL_RUN = run_id
        MODEL_RUNS[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "mode": "actual local model training",
            "dataset_id": config.dataset_id,
            "dataset": DATASET_LABELS.get(config.dataset_id, config.dataset_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rounds_total": config.rounds,
            "rounds_completed": 0,
            "config": config.model_dump(),
            "output_dir": str(output_dir.relative_to(ROOT)),
            "history": [],
            "notice": "Measured results from your local dataset. This run does not upload image files; the aggregation server sees client updates before aggregate noise is added.",
        }
    background_tasks.add_task(_run_model_process, run_id, config, output_dir, data_root)
    return {"run_id": run_id, "status": "queued", "status_url": f"/api/model/run/{run_id}"}


@app.get("/api/model/run/{run_id}")
async def get_model_run(run_id: str) -> dict[str, Any]:
    with MODEL_RUN_LOCK:
        run = MODEL_RUNS.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Model run not found")
        snapshot = dict(run)
    metrics_file = ROOT / snapshot["output_dir"] / "metrics.csv"
    if metrics_file.is_file():
        with metrics_file.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        history = []
        for row in rows:
            selected_ids = [value for value in row.get("selected_client_ids", "").split(",") if value]
            round_number = int(row["round"])
            history.append({
                "round": round_number,
                "progress": round(round_number * 100 / snapshot["rounds_total"]),
                "sampled_clients": selected_ids,
                "client_count": int(row["sampled_clients"]),
                "train_loss": float(row["test_loss"]),
                "validation_accuracy": float(row["test_accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "epsilon": row["epsilon"],
                "delta": row["delta"],
                "raw_images_transferred": 0,
                "protection": "DP noise enabled" if float(snapshot["config"]["noise_multiplier"]) > 0 else "privacy disabled",
            })
        snapshot["history"] = history
        snapshot["rounds_completed"] = len(history)
        snapshot["latest"] = history[-1] if history else None
    return snapshot


# Keep this last so /api routes take precedence. Relative asset links also let
# frontend/index.html work when opened directly as a file:// preview.
app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")

