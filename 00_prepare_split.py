#!/usr/bin/env python3
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    CLAHE_DIR,
    CLAHE_ZIP,
    CLASS_NAMES,
    DATASET_DIR,
    IMAGE_EXTS,
    SEED,
    SPLIT_MANIFEST,
    BATCH_SIZE,
    YOLO_EPOCHS,
    YOLO_PATIENCE,
    CNN_EPOCHS,
    CNN_PATIENCE,
    WORKING,
)


def find_source_root(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None

    candidates = []
    for parasitized in search_root.rglob("Parasitized"):
        if not parasitized.is_dir():
            continue
        uninfected = parasitized.parent / "Uninfected"
        if uninfected.is_dir():
            candidates.append(parasitized.parent)

    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(path.parts), str(path)))


def extract_clahe_archive() -> Path | None:
    if CLAHE_DIR.is_dir():
        return CLAHE_DIR
    if not CLAHE_ZIP.exists():
        return None

    print(f"Mengekstrak {CLAHE_ZIP.name}...")
    with zipfile.ZipFile(CLAHE_ZIP) as archive:
        archive.extractall(CLAHE_DIR.parent)
    return CLAHE_DIR if CLAHE_DIR.is_dir() else None


def build_split(source_root: Path) -> pd.DataFrame:
    source_files = {}
    for class_name in CLASS_NAMES:
        class_dir = source_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

        files = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ),
            key=lambda path: str(path),
        )
        source_files[class_name] = files
        print(f"{class_name}: {len(files):,} gambar")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    rows = []
    for class_name, paths in source_files.items():
        train_files, rest = train_test_split(
            paths, test_size=0.20, random_state=SEED, shuffle=True
        )
        val_files, test_files = train_test_split(
            rest, test_size=0.50, random_state=SEED, shuffle=True
        )

        for split_name, split_files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            split_dir = DATASET_DIR / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(split_files):
                destination = split_dir / f"{index:06d}_{source.name}"
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)

                rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "image_path": str(destination),
                        "source_path": str(source),
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(SPLIT_MANIFEST, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan split 80/10/10")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    split_ready = (
        not args.rebuild
        and SPLIT_MANIFEST.exists()
        and (DATASET_DIR / "test").is_dir()
    )
    if split_ready:
        print(f"Split sudah ada: {SPLIT_MANIFEST}")
        manifest = pd.read_csv(SPLIT_MANIFEST)
    else:
        source_root = args.source
        if source_root is not None:
            source_root = find_source_root(source_root) or source_root
        else:
            source_root = extract_clahe_archive()
            if source_root is None:
                source_root = find_source_root(Path("/kaggle/input"))

        if source_root is None:
            raise FileNotFoundError(
                "Dataset tidak ditemukan. Gunakan --source atau letakkan "
                "zip CLAHE/ folder clahe di folder project."
            )

        print(f"Source root: {source_root}")
        manifest = build_split(source_root)

    print(pd.crosstab(manifest["split"], manifest["class_name"]))
    print(f"Total gambar: {len(manifest):,}")
    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "split": {"train": 22046, "val": 2756, "test": 2756},
        "yolo": {
            "epochs": YOLO_EPOCHS,
            "patience": YOLO_PATIENCE,
            "image_size": 224,
            "best_model_metric": "validation_loss",
        },
        "cnn": {
            "epochs": CNN_EPOCHS,
            "patience": CNN_PATIENCE,
            "image_size": 224,
            "best_model_metric": "validation_loss",
        },
        "svm": {
            "pixel_size": 32,
            "grid_subsample": 8000,
            "best_model_metric": "validation_hinge_loss",
        },
    }
    (WORKING / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

