"""Small, readable starter for client-level DP FedAvg with a Deep CapsNet.

Expected data layout (one folder per simulated institution):
data/clients/client_0/<class_name>/*.{png,jpg,...}
data/clients/client_1/<class_name>/*.{png,jpg,...}
data/test/<class_name>/*.{png,jpg,...}

Client sampling is Poisson. Each sampled client's example-count-weighted model
delta is L2-clipped; Gaussian noise is added to the sum before the server
applies the update. Opacus RDP accounting reports an approximate client-level
(epsilon, delta) guarantee under the stated sampling and adjacency assumptions.
This is an experimental starter, not a clinical or production deployment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus.accountants import RDPAccountant


class PrimaryCapsules(nn.Module):
    """Convolutional features reshaped into spatial capsule vectors."""

    def __init__(self, in_channels: int, capsule_types: int = 8, capsule_dim: int = 8):
        super().__init__()
        self.capsule_types = capsule_types
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, capsule_types * capsule_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        batch, _, height, width = x.shape
        x = x.view(batch, self.capsule_types, self.capsule_dim, height, width)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return squash(x.view(batch, height * width * self.capsule_types, self.capsule_dim))


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm_sq = (x * x).sum(dim=dim, keepdim=True)
    scale = norm_sq / (1.0 + norm_sq) / torch.sqrt(norm_sq + 1e-8)
    return scale * x


class DigitCapsules(nn.Module):
    """Fully connected output capsules with dynamic routing by agreement."""

    def __init__(self, input_capsules: int, input_dim: int, classes: int, output_dim: int = 16, routing_iters: int = 3):
        super().__init__()
        self.classes = classes
        self.routing_iters = routing_iters
        self.weight = nn.Parameter(torch.empty(1, input_capsules, classes, output_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Votes: [batch, input capsule, output class, output dimension]
        votes = torch.einsum("bid,iohd->bioh", x, self.weight.squeeze(0))
        logits = votes.new_zeros(votes.shape[0], votes.shape[1], self.classes, 1)
        for iteration in range(self.routing_iters):
            coupling = torch.softmax(logits, dim=2)
            output = squash((coupling * votes).sum(dim=1))
            if iteration < self.routing_iters - 1:
                agreement = (votes * output.unsqueeze(1)).sum(dim=-1, keepdim=True)
                logits = logits + agreement
        return output


class DeepCapsNet(nn.Module):
    """Compact convolutional capsule classifier for 1- or 3-channel images."""

    def __init__(self, in_channels: int, classes: int, routing_iters: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
        )
        self.primary = PrimaryCapsules(128)
        self.digit = DigitCapsules(input_capsules=4 * 4 * 8, input_dim=8, classes=classes, routing_iters=routing_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capsules = self.digit(self.primary(self.features(x)))
        return torch.linalg.vector_norm(capsules, dim=-1)


class MarginLoss(nn.Module):
    """CapsNet margin loss; labels are integer class IDs."""

    def __init__(self, positive_margin: float = 0.9, negative_margin: float = 0.1, negative_weight: float = 0.5):
        super().__init__()
        self.m_plus = positive_margin
        self.m_minus = negative_margin
        self.negative_weight = negative_weight

    def forward(self, class_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = torch.nn.functional.one_hot(labels, num_classes=class_scores.shape[1]).to(class_scores.dtype)
        positive = targets * torch.relu(self.m_plus - class_scores).square()
        negative = (1.0 - targets) * torch.relu(class_scores - self.m_minus).square()
        return (positive + self.negative_weight * negative).sum(dim=1).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Deep CapsNet with client-level DP FedAvg.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--client-sampling-rate", type=float, default=0.5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--routing-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(path: Path, classes: List[str], transform: transforms.Compose) -> datasets.ImageFolder:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing image folder: {path}")
    dataset = datasets.ImageFolder(path, transform=transform)
    if dataset.classes != classes:
        raise ValueError(f"Class folders in {path} are {dataset.classes}; expected {classes}.")
    return dataset


def state_delta(local_state: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: local_state[key] - global_state[key] for key in global_state}


def clip_weighted_delta(delta: Dict[str, torch.Tensor], client_weight: float, max_norm: float) -> Tuple[Dict[str, torch.Tensor], float]:
    weighted = {key: value * client_weight for key, value in delta.items()}
    norm = torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in weighted.values()]))
    factor = min(1.0, max_norm / (norm.item() + 1e-12))
    return {key: value * factor for key, value in weighted.items()}, norm.item()


def train_client(
    global_state: Dict[str, torch.Tensor], dataset: datasets.ImageFolder, model_factory, device: torch.device,
    batch_size: int, epochs: int, learning_rate: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    model = model_factory().to(device)
    model.load_state_dict(global_state)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    loss_fn = MarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss, batches = 0.0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
    return copy.deepcopy(model.state_dict()), total_loss / max(1, batches)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        total_loss += loss_fn(scores, labels).item() * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / max(1, count), correct / max(1, count)


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("--rounds and --local-epochs must be at least 1.")
    if not (0 < args.client_sampling_rate <= 1):
        raise ValueError("--client-sampling-rate must be in (0, 1].")
    if args.clip_norm <= 0 or args.noise_multiplier < 0 or not (0 < args.delta < 1):
        raise ValueError("Use clip-norm > 0, noise-multiplier >= 0, and delta in (0, 1).")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_root = args.data_dir / "clients"
    client_paths = sorted(path for path in client_root.glob("client_*") if path.is_dir())
    if len(client_paths) < 2:
        raise ValueError(f"Create at least two client folders under {client_root}.")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1) if args.channels == 1 else transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * args.channels, std=[0.5] * args.channels),
    ])
    first_dataset = datasets.ImageFolder(client_paths[0], transform=transform)
    classes = first_dataset.classes
    if len(classes) < 2:
        raise ValueError("Classification needs at least two class subfolders.")
    client_datasets = [first_dataset] + [make_dataset(path, classes, transform) for path in client_paths[1:]]
    if any(len(dataset) == 0 for dataset in client_datasets):
        raise ValueError("Every client folder must contain at least one image.")
    test_dataset = make_dataset(args.data_dir / "test", classes, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    total_examples = sum(len(dataset) for dataset in client_datasets)

    def model_factory():
        return DeepCapsNet(args.channels, len(classes), args.routing_iters)

    global_model = model_factory().to(device)
    global_state = copy.deepcopy(global_model.state_dict())
    accountant = RDPAccountant()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["round", "sampled_clients", "mean_client_loss", "test_loss", "test_accuracy", "epsilon", "delta"])
        writer.writeheader()
        for round_index in range(1, args.rounds + 1):
            selected = [i for i in range(len(client_datasets)) if random.random() < args.client_sampling_rate]
            aggregate = {key: torch.zeros_like(value) for key, value in global_state.items()}
            client_losses: List[float] = []
            for client_index in selected:
                dataset = client_datasets[client_index]
                local_state, local_loss = train_client(
                    global_state, dataset, model_factory, device,
                    args.batch_size, args.local_epochs, args.learning_rate,
                )
                client_weight = len(dataset) / total_examples
                contribution, _ = clip_weighted_delta(state_delta(local_state, global_state), client_weight, args.clip_norm)
                for key in aggregate:
                    aggregate[key] += contribution[key]
                client_losses.append(local_loss)

            # Gaussian noise is applied to the clipped sum, before the server update.
            if selected and args.noise_multiplier > 0:
                for key in aggregate:
                    aggregate[key] += torch.randn_like(aggregate[key]) * (args.noise_multiplier * args.clip_norm)
            for key in global_state:
                global_state[key] += aggregate[key]
            # RDPAccountant uses Poisson subsampling; the same rate is used above.
            accountant.step(noise_multiplier=max(args.noise_multiplier, 1e-12), sample_rate=args.client_sampling_rate)
            epsilon = accountant.get_epsilon(delta=args.delta) if args.noise_multiplier > 0 else math.inf
            global_model.load_state_dict(global_state)
            test_loss, test_accuracy = evaluate(global_model, test_loader, device, MarginLoss())
            row = {
                "round": round_index, "sampled_clients": len(selected),
                "mean_client_loss": f"{sum(client_losses) / len(client_losses):.6f}" if client_losses else "",
                "test_loss": f"{test_loss:.6f}", "test_accuracy": f"{test_accuracy:.6f}",
                "epsilon": f"{epsilon:.6f}" if math.isfinite(epsilon) else "infinite", "delta": args.delta,
            }
            writer.writerow(row)
            metrics_file.flush()
            print(f"Round {round_index:03d}/{args.rounds} | clients={len(selected)} | test acc={test_accuracy:.4f} | epsilon={row['epsilon']}")

    global_model.load_state_dict(global_state)
    torch.save({"model_state_dict": global_state, "classes": classes, "channels": args.channels, "image_size": args.image_size}, args.output_dir / "deep_capsnet_fedavg_dp.pt")
    privacy_report = {
        "privacy_unit": "client",
        "mechanism": "Poisson-sampled client-level Gaussian DP-FedAvg",
        "rounds": args.rounds,
        "client_count": len(client_datasets),
        "client_sampling_rate": args.client_sampling_rate,
        "clip_norm": args.clip_norm,
        "noise_multiplier": args.noise_multiplier,
        "delta": args.delta,
        "epsilon": accountant.get_epsilon(delta=args.delta) if args.noise_multiplier > 0 else None,
        "accountant": "Opacus RDPAccountant",
        "notes": "Each client’s example-count-weighted model delta is clipped as one contribution. The Gaussian mechanism is applied to the clipped sum. Guarantee depends on client-level adjacency, Poisson sampling, trusted server execution, and correct configuration. This does not provide patient-level DP within a client.",
    }
    (args.output_dir / "privacy_report.json").write_text(json.dumps(privacy_report, indent=2), encoding="utf-8")
    print(f"Saved model, metrics, and privacy report in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
