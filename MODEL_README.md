# MedFedAI model starter

This starter trains a compact Deep Capsule Network for image classification using simulated hospital clients. Clients train locally; the server receives model deltas and applies FedAvg. Client-level differential privacy clips each client's weighted delta and adds Gaussian noise to the aggregate.

## 1. Prepare the data

Split images into client training folders and a separate held-out test folder. Use the same class folder names under every client and test directory:

```text
data/
  clients/
    client_0/
      class_a/ image1.png ...
      class_b/ image2.png ...
    client_1/
      class_a/ ...
      class_b/ ...
  test/
    class_a/ ...
    class_b/ ...
```

The code reads each client's files from its folder. It does not copy images into a server folder. This starter does not split patient records or datasets automatically. Make the train/test split at the patient level before arranging files to reduce identity leakage across the split. Do not use a test set for local training.

## 2. Install dependencies

Use Python 3.10 or newer in a virtual environment. Install a PyTorch build appropriate for your CPU/GPU, then install the remaining packages:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run a small experiment

```bash
python train_federated.py --data-dir data --rounds 20 --local-epochs 1 --batch-size 8 --client-sampling-rate 0.5 --clip-norm 1.0 --noise-multiplier 1.0 --delta 1e-5
```

Start with a small number of images, clients, and rounds to confirm the data layout and memory use. Use `--channels 1` for grayscale images and `--channels 3` for RGB images. Adjust `--image-size` for the data and available memory.

## 4. Outputs

The `outputs/` directory contains:

- `deep_capsnet_fedavg_dp.pt` — final weights, class labels, and image settings.
- `metrics.csv` — per-round client count, losses, test accuracy, epsilon, and delta.
- `privacy_report.json` — mechanism configuration and reported privacy budget.

The model uses convolutional feature layers, primary capsules, dynamic routing, and margin loss. The client sampler uses independent Poisson selection with rate `q`; the RDP accountant uses that same rate. The clipped contribution is the client's example-count-weighted delta, and noise standard deviation on the summed update is `noise_multiplier × clip_norm`.

## Privacy and research limitations

- This is **central client-level DP**: the aggregation server sees client updates before adding noise and must be trusted. It is not secure aggregation and does not hide updates from the server.
- It protects the participation/contribution of a whole client under the implemented adjacency assumptions. It does **not** provide patient-level DP within each hospital.
- The accountant assumes Poisson client sampling and that client dataset sizes used for FedAvg weighting are public. Record these assumptions and validate them for the experiment protocol.
- Setting `--noise-multiplier 0` disables noise and the reported epsilon is infinite; do not describe that run as differentially private.
- Differential privacy trades model utility for privacy. Compare baseline and protected runs on the same split and report accuracy, privacy settings, random seeds, and compute limits.
- This code is a research prototype, not a medical device or clinical decision system. The controlled privacy attack and its evaluation protocol still need to be implemented as a separate experiment.
