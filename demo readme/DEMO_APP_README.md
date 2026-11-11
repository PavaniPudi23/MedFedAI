# MedFedAI review demo: frontend + backend

This app gives you a local dashboard to present the project flow: simulated healthcare clients, Deep CapsNet components, FedAvg rounds, and differential privacy controls. It runs in VS Code, Antigravity, or any terminal with Python.

## Important demo boundary

The dashboard backend simulates round-by-round metrics so the interface can be demonstrated before the medical dataset and full training pipeline are ready. **It does not train a neural network, read images, send model updates, apply Gaussian noise, or calculate a privacy guarantee.** The page labels these metrics as illustrative. The separate `train_federated.py` script is the actual PyTorch training prototype; see `MODEL_README.md` for its data layout and privacy assumptions.

## Run the website locally

Open this project folder in Visual Studio Code or Antigravity, then open its integrated terminal in the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-web.txt
python -m uvicorn backend.main:app --reload
```

If PowerShell blocks virtual-environment activation, run the commands below without activation instead:

```powershell
python -m pip install -r requirements-web.txt
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser. Use **Run demo experiment** to animate federated rounds. The backend API is documented at `http://127.0.0.1:8000/docs`.

You can also double-click `frontend/index.html` for a styled, browser-only preview. It will show **Preview mode · API offline** and the demo button will run a local illustration. Use the server instructions above when you want the frontend connected to the FastAPI backend.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
python -m uvicorn backend.main:app --reload
```

## App files

- `backend/main.py` — FastAPI endpoints for health, overview, start demo, and run status.
- `frontend/index.html` — dashboard structure.
- `frontend/dashboard.css` — responsive styling.
- `frontend/dashboard.js` — API calls, simulated live progress, and chart.
- `train_federated.py` — separate real model training prototype.

## For your project review

Describe the dashboard as a **frontend/backend workflow prototype** and the results as simulated. You can demonstrate the architecture and user flow now. For measured classification accuracy or a DP guarantee, run the training script on a properly split labeled dataset and show those experiment outputs separately.
