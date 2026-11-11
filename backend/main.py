"""MedFedAI review demo API.

This backend simulates federated rounds for a UI demonstration. It does not load
medical images, train weights, or provide a differential privacy guarantee.
Use train_federated.py for the separate PyTorch model training prototype.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
app = FastAPI(title="MedFedAI Demo API", version="0.1.0")
RUNS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()

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


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "review-simulation"}


@app.get("/api/overview")
async def overview() -> dict[str, Any]:
    return {
        "project": "MedFedAI",
        "mode": "simulated review demo",
        "clients": CLIENTS,
        "model": {"name": "Deep Capsule Network", "features": ["Convolutional feature layers", "Primary capsules", "Dynamic routing", "Margin loss"]},
        "federated_learning": "Clients train locally; FedAvg combines model updates at the server.",
        "privacy": {"mechanism": "Differential privacy concept demo", "controls": ["Client update clipping", "Gaussian noise setting"]},
        "notice": "Illustrative simulation only. No images or model weights are processed by this demo API.",
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


# Keep this last so /api routes take precedence. Relative asset links also let
# frontend/index.html work when opened directly as a file:// preview.
app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")

