"""Create a deterministic train/test split and simulated IID client folders."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simulated client folders from a class-folder image dataset.")
    parser.add_argument("--source", type=Path, required=True, help="Training/source folder containing one subfolder per class")
    parser.add_argument("--test-source", type=Path, help="Optional existing held-out test split; copied unchanged by class")
    parser.add_argument("--validation-source", type=Path, help="Optional validation split to preserve under output/validation")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output folder (must not already contain files)")
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy", help="Copy images or make same-drive hard links to avoid duplicating image storage")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source class folder not found: {args.source}")
    if args.clients < 2:
        raise ValueError("Use at least two simulated clients.")
    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.05 and 0.5.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.output}. Choose a new output path.")

    class_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Source must contain at least two class subfolders.")
    class_names = [path.name for path in class_dirs]
    for split_name, split_path in (("test", args.test_source), ("validation", args.validation_source)):
        if split_path is None:
            continue
        if not split_path.is_dir():
            raise FileNotFoundError(f"{split_name.title()} split not found: {split_path}")
        split_classes = sorted(path.name for path in split_path.iterdir() if path.is_dir())
        if split_classes != class_names:
            raise ValueError(f"Class folders in {split_name} split ({split_classes}) do not match source ({class_names}).")
    rng = random.Random(args.seed)
    client_counts: list[Counter[str]] = [Counter() for _ in range(args.clients)]
    test_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()

    def place_image(source_image: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.link_mode == "hardlink":
            try:
                os.link(source_image, destination)
            except OSError as error:
                raise OSError(
                    f"Could not hard-link {source_image} to {destination}. "
                    "Hard links require source and output on the same drive and filesystem; "
                    "use --link-mode copy if that is not possible."
                ) from error
        else:
            shutil.copy2(source_image, destination)

    for class_dir in class_dirs:
        images = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < args.clients + 1:
            raise ValueError(f"Class '{class_dir.name}' needs at least {args.clients + 1} images; found {len(images)}.")
        rng.shuffle(images)
        if args.test_source:
            train_images = images
            test_dir = args.test_source / class_dir.name
            test_images = sorted(path for path in test_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if not test_images:
                raise ValueError(f"No images found for class '{class_dir.name}' in {args.test_source}.")
        else:
            test_count = max(1, round(len(images) * args.test_fraction))
            test_images = images[:test_count]
            train_images = images[test_count:]

        for image_index, source_image in enumerate(test_images):
            destination = args.output / "test" / class_dir.name / f"{image_index:05d}_{source_image.name}"
            place_image(source_image, destination)
            test_counts[class_dir.name] += 1

        if args.validation_source:
            validation_dir = args.validation_source / class_dir.name
            validation_images = sorted(path for path in validation_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            for image_index, source_image in enumerate(validation_images):
                destination = args.output / "validation" / class_dir.name / f"{image_index:05d}_{source_image.name}"
                place_image(source_image, destination)
                validation_counts[class_dir.name] += 1

        # Round-robin each shuffled class across clients so all clients have a
        # similar class mix for this initial IID experiment.
        for image_index, source_image in enumerate(train_images):
            client_index = image_index % args.clients
            destination = args.output / "clients" / f"client_{client_index}" / class_dir.name / f"{image_index:05d}_{source_image.name}"
            place_image(source_image, destination)
            client_counts[client_index][class_dir.name] += 1

    report = {
        "source": str(args.source.resolve()),
        "test_source": str(args.test_source.resolve()) if args.test_source else None,
        "validation_source": str(args.validation_source.resolve()) if args.validation_source else None,
        "split_type": "official test split preserved; source training images distributed approximately IID" if args.test_source else "stratified image-level train/test split; approximately IID client allocation",
        "seed": args.seed,
        "test_fraction": args.test_fraction if not args.test_source else None,
        "client_count": args.clients,
        "client_class_counts": {f"client_{index}": dict(counts) for index, counts in enumerate(client_counts)},
        "test_class_counts": dict(test_counts),
        "validation_class_counts": dict(validation_counts),
        "patient_level_split": False,
        "image_storage_mode": args.link_mode,
        "warning": "Client distribution is approximately IID. Patient-level separation is not verified because patient identifiers were unavailable.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {args.clients} client folders and a held-out test folder at {args.output.resolve()} ({args.link_mode} mode)")
    print(json.dumps({"client_class_counts": report["client_class_counts"], "test_class_counts": report["test_class_counts"], "validation_class_counts": report["validation_class_counts"]}, indent=2))
    print("Note: patient-level separation is not verified because patient identifiers were unavailable.")


if __name__ == "__main__":
    main()
