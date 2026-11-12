# MedFedAI local model dashboard

The dashboard lets you import your own labeled image folder and starts the actual PyTorch Deep CapsNet training program locally. Imported images are sent only to the MedFedAI backend on this computer (`127.0.0.1`) for local preparation; no remote service receives them.

## Data and privacy boundary

The importer creates a local `data_manual_*` folder. A class-folder-only dataset gets an image-level 80/20 split; a supplied `train/` and `test/` split is preserved. The training images are assigned to the number of simulated clients you choose. During training, the aggregation process sees individual client updates before aggregate noise is added; this prototype assumes a trusted server and reports **client-level**, not patient-level, differential privacy. Do not expose this local training server publicly.

Imported data, experiment outputs, and source dataset folders are excluded from Git. Import and select your dataset from the Overview page before training.

## Run the local website

Open this project folder in Visual Studio Code or Antigravity, then open an integrated terminal in the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision
python -m pip install -r "demo readme\requirements.txt"
python -m pip install -r "demo readme\requirements-web.txt"
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`, import a labeled dataset folder, select it, and choose **Train selected dataset**. Set the noise multiplier to `0` for an unprotected baseline or a positive value for the DP run. Each experiment saves a checkpoint, `metrics.csv`, and `privacy_report.json` under `outputs/`. API documentation is at `http://127.0.0.1:8000/docs`.

Opening `frontend/index.html` directly does not start training. Use the local FastAPI server so the dashboard can reach the training backend.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision
python -m pip install -r "demo readme/requirements.txt"
python -m pip install -r "demo readme/requirements-web.txt"
python -m uvicorn backend.main:app --reload
```

## Main files

- `backend/main.py` — FastAPI endpoints for local training and run status.
- `frontend/index.html` — dashboard layout and controls.
- `frontend/dashboard.css` — responsive styling.
- `frontend/dashboard.js` — starts training and displays measured metrics.
- `demo readme/train_federated.py` — Deep CapsNet, FedAvg, clipping, Gaussian noise, and accounting.
- `demo readme/predict_image.py` — single-image inference using a saved checkpoint.

## For the project review

Report the dataset split, client-level privacy unit, noise and clipping settings, epsilon and delta, measured metrics, and limitations. This small simulated-client system is a research prototype, not a real multi-hospital deployment or clinical tool.
