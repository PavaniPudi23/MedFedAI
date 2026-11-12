# MedFedAI: runnable model prototype

This folder contains an actual PyTorch training program for a compact Deep Capsule Network (Deep CapsNet). It trains simulated clients on their own image folders, combines their updates with FedAvg, and can add Gaussian noise for a **central client-level differential-privacy experiment**. It reports measured test metrics from the dataset you provide; it does not make up accuracy values.

The website dashboard in `frontend/` is still a visual simulation. Running that dashboard does not start this model. Use the commands below to run actual training locally.

## 1. Arrange a labelled image dataset

Make a patient-independent train/test split first, if patient identifiers are available. Then arrange training images into one folder per simulated client. Use the same class-folder names and spelling for every client and for the test set:

```text
Capstone Project/
  data/
    clients/
      client_0/
        class_a/image001.png
        class_b/image002.png
      client_1/
        class_a/image003.png
        class_b/image004.png
    test/
      class_a/image101.png
      class_b/image102.png
  demo readme/
    train_federated.py
```

For example, the folders could be `normal/` and `pneumonia/`. Do not put test images inside a client folder. The current program expects images in ordinary class subfolders and supports common formats handled by Pillow. It does not download a dataset.

For the knee dataset currently in this workspace, create the local split once from the **MedicalExpert-I** labels:

```powershell
python "demo readme\prepare_imagefolder_clients.py" --source "Knee - Dataset\MedicalExpert-I" --output data --clients 3 --test-fraction 0.2 --seed 17
```

The script makes a stratified image-level 80/20 split and distributes training images approximately IID across three clients. It does not use MedicalExpert-II as a separate client because the expert folders contain overlapping image files with potentially different labels. The generated `data/` folder is ignored by Git and stays local. Patient IDs were not available for this split, so it cannot guarantee that images from one patient stay in only one partition.

### Optional chest X-ray dataset

The `chest x-ray/chest_xray/chest_xray/` dataset already has `train`, `val`, and `test` folders. Preserve those official splits and build a separate local client dataset with:

```powershell
python "demo readme\prepare_imagefolder_clients.py" --source "chest x-ray\chest_xray\chest_xray\train" --test-source "chest x-ray\chest_xray\chest_xray\test" --validation-source "chest x-ray\chest_xray\chest_xray\val" --output data_chest --clients 3 --seed 17 --link-mode hardlink
```

Hard links avoid storing another full copy of the images, but require the source and output to be on the same drive. If hard links are unavailable, rerun with `--link-mode copy` (this uses extra disk space). The source dataset and generated `data_chest/` are ignored by Git so the images are not committed.

The dashboard is set up for user-imported datasets. In **Overview → Import images from your computer**, select a folder containing either class folders (for example `NORMAL/image.jpg`, `PNEUMONIA/image.jpg`) or split folders (`train/<class>/`, `test/<class>/`, and optional `val/<class>/`). The importer creates an image-level 80/20 test split for class-folder-only datasets, or preserves an included train/test split. It distributes the training images across the client count you choose, then lists the imported dataset in the selector. The same selection is used for training, held-out evaluation, and checkpoints. Imported folders are stored locally under `data_manual_*` and ignored by Git. Validation images are retained for future use; the current trainer evaluates on the held-out test set.

## 2. Install the model dependencies

Open PowerShell in the project root (the folder containing `data` and `demo readme`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r "demo readme\requirements.txt"
```

If your computer has no supported GPU, PyTorch will use the CPU. CPU runs may take longer, so begin with a small dataset, fewer rounds, and a small image size.

## 3. Run the unprotected baseline

From the project root:

```powershell
python "demo readme\train_federated.py" --data-dir data --output-dir outputs\baseline --rounds 3 --local-epochs 1 --batch-size 4 --image-size 32 --client-sampling-rate 1.0 --noise-multiplier 0 --seed 17
```

Here, `noise-multiplier 0` means the DP noise is disabled. This run is a baseline and must not be described as private.

## 4. Run with client-level differential privacy

Use the same dataset split, model settings, and seed for a fair comparison. Begin with a short experiment, then increase rounds when it runs within your computer's limits:

```powershell
python "demo readme\train_federated.py" --data-dir data --output-dir outputs\dp --rounds 3 --local-epochs 1 --batch-size 4 --image-size 32 --client-sampling-rate 1.0 --clip-norm 1.0 --noise-multiplier 1.0 --delta 1e-5 --seed 17
```

Use `--channels 1` for grayscale images or `--channels 3` for RGB images. For initial debugging, use `--client-sampling-rate 1.0` so every client participates each round; lower sampling rates are available for experiments with enough clients.

## What the training program does

1. Loads the class folders separately for each simulated client; the central evaluator loads only the held-out test folder.
2. Trains a Deep CapsNet locally at each selected client using convolutional layers, primary capsules, dynamic routing, and margin loss.
3. Forms each client's example-count-weighted model update, clips its L2 norm, and adds calibrated Gaussian noise to the sampled aggregate.
4. Uses FedAvg to update the global model, then measures test loss, accuracy, macro precision, macro recall, and macro F1.
5. Tracks epsilon and delta with the Opacus Rényi Differential Privacy accountant for runs where noise is enabled.

## Output files

Each output directory contains:

- `deep_capsnet_fedavg_dp.pt` — trained model weights, class names, and image/model settings.
- `metrics.csv` — measured metrics and privacy accounting by federated round.
- `privacy_report.json` — privacy mechanism settings and accounting assumptions.

## Classify one image with the saved model

After a training run, use the matching checkpoint and input channel setting (the channel count and image size are stored in the checkpoint):

```powershell
python "demo readme\predict_image.py" --checkpoint outputs\dp\deep_capsnet_fedavg_dp.pt --image path\to\image.png
```

The script prints the predicted class and capsule scores. These scores are not calibrated probabilities, and the prediction is for demonstration/research only—not medical advice.

## View the model phases in the dashboard

After at least one checkpoint exists under `outputs/`, open the local dashboard, enter a held-out test-image number, and choose **Show image phases**. The dashboard displays the original image, its preprocessed version, a convolution activation heatmap, a primary-capsule activity map, and the class-capsule scores after dynamic routing. The backend reads the sample from the local test folder and serves the visualization to the local browser; no image-upload control is used. Heatmaps show model activations, not medical image reconstructions, and capsule scores are not calibrated probabilities.

## Privacy boundary and limits

- This implementation applies **central client-level DP** to the aggregated client contributions. The aggregation server sees the individual client updates before noise is added and must be trusted. This is a privacy experiment, not secure aggregation or general cybersecurity.
- The privacy unit is a whole simulated client. This does **not** guarantee that an individual patient's image is protected from that client's local training. Do not call it patient-level privacy.
- The accountant assumes Poisson client sampling and public client dataset sizes used for FedAvg weighting. Record these assumptions with results.
- The reported epsilon is meaningful only for a correctly configured run with positive noise. With `--noise-multiplier 0`, privacy is disabled and epsilon is infinite.
- Differential privacy may reduce classification performance. Report both the baseline and protected metrics; do not claim that privacy has no effect.
- Client data can be non-IID (different class proportions or image sources). This implementation does not create a particular non-IID partition; arrange client folders to match the experiment you intend to study and document that distribution.
- These results are for a research prototype and are not clinical diagnoses or a medical device.
