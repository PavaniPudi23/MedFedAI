"""Run one local image through a trained MedFedAI Deep CapsNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_federated import DeepCapsNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image with a saved Deep CapsNet model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to deep_capsnet_fedavg_dp.pt")
    parser.add_argument("--image", type=Path, required=True, help="Path to a local image file")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    channels = int(checkpoint["channels"])
    image_size = int(checkpoint["image_size"])
    routing_iters = int(checkpoint.get("routing_iters", 3))
    model = DeepCapsNet(channels, len(classes), routing_iters).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1) if channels == 1 else transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ])
    with Image.open(args.image) as source:
        image = image_transform(source).unsqueeze(0).to(device)
    with torch.no_grad():
        class_scores = model(image)[0]
    predicted_index = int(class_scores.argmax().item())
    print(f"Predicted class: {classes[predicted_index]}")
    print("Capsule class scores (not calibrated probabilities):")
    for index in class_scores.argsort(descending=True).tolist():
        print(f"  {classes[index]}: {class_scores[index].item():.4f}")
    print("Research prototype only; this output is not a medical diagnosis.")


if __name__ == "__main__":
    main()
