#!/usr/bin/env python3
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import (
    BATCH_SIZE,
    CNN_DIR,
    CNN_EPOCHS,
    CNN_PATIENCE,
    DATASET_DIR,
    SEED,
    WORKING,
)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = block(3, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, inputs):
        outputs = self.block3(self.block2(self.block1(inputs)))
        outputs = self.global_avg_pool(outputs)
        return self.fc(torch.flatten(outputs, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Custom CNN")
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument("--patience", type=int, default=CNN_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CNN_DIR.parent / "custom_cnn_lr5e4_150ep",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm

    cnn_output_dir = args.output_dir

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "PyTorch:",
        torch.__version__,
        "| Device:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    )
    if not DATASET_DIR.exists():
        raise FileNotFoundError("Split belum ada. Jalankan 00_prepare_split.py.")

    cnn_output_dir.mkdir(parents=True, exist_ok=True)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test", transform=eval_transform
    )
    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise RuntimeError("Mapping kelas antar split tidak sama.")

    generator = torch.Generator().manual_seed(SEED)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    cnn = CustomCNN(num_classes=2).to(device)
    print(
        "Kelas:",
        train_dataset.classes,
        "| Train/Val/Test:",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    print("Total parameter:", f"{sum(p.numel() for p in cnn.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        cnn.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = cnn_output_dir / "best_custom_cnn.pth"
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    for epoch in range(args.epochs):
        cnn.train()
        train_loss = train_correct = train_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = cnn(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        cnn.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = cnn(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["lr"].append(learning_rate)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | LR {learning_rate:.1e} "
            f"| Train Loss {train_loss:.4f} Acc {train_accuracy:.4f} "
            f"| Val Loss {val_loss:.4f} Acc {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": cnn.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                best_path,
            )
            print("  -> best model saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    history_frame = pd.DataFrame(history)
    history_frame.index = history_frame.index + 1
    history_frame.index.name = "epoch"
    history_frame.to_csv(cnn_output_dir / "training_history.csv")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    true_values = []
    predicted_values = []
    probability_values = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing CNN"):
            probabilities = torch.softmax(cnn(images.to(device)), dim=1)
            true_values.extend(labels.numpy())
            predicted_values.extend(probabilities.argmax(1).cpu().numpy())
            probability_values.extend(probabilities.cpu().numpy())

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    probability_values = np.array(probability_values)
    positive_index = train_dataset.class_to_idx["Parasitized"]
    binary_true = (true_values == positive_index).astype(int)
    binary_predicted = (predicted_values == positive_index).astype(int)
    scores = probability_values[:, positive_index]

    metrics = {
        "model": "Custom CNN",
        "n_test": int(len(binary_true)),
        "accuracy": float(accuracy_score(binary_true, binary_predicted)),
        "precision": float(
            precision_score(binary_true, binary_predicted, zero_division=0)
        ),
        "recall": float(recall_score(binary_true, binary_predicted, zero_division=0)),
        "f1": float(f1_score(binary_true, binary_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(binary_true, scores)),
    }
    pd.DataFrame([metrics]).to_csv(
        cnn_output_dir / "test_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "image_path": [str(path) for path, _ in test_dataset.samples],
            "y": binary_true,
            "pred": binary_predicted,
            "score": scores,
        }
    ).to_csv(cnn_output_dir / "cnn_test_predictions.csv", index=False)

    print("=== CUSTOM CNN TEST ===")
    for key, value in metrics.items():
        print(f"{key:<22}: {value}")
    print(
        classification_report(
            binary_true,
            binary_predicted,
            target_names=["Uninfected", "Parasitized"],
            digits=4,
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    matrix = confusion_matrix(binary_true, binary_predicted, labels=[1, 0])
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Parasitized", "Uninfected"],
        yticklabels=["Parasitized", "Uninfected"],
    )
    axes[0].set_title("Custom CNN - Confusion Matrix")

    false_positive, true_positive, _ = roc_curve(binary_true, scores)
    axes[1].plot(
        false_positive,
        true_positive,
        label=f"AUC = {metrics['roc_auc']:.4f}",
    )
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Custom CNN - ROC")
    axes[1].legend()
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig(
        cnn_output_dir / "cnn_confusion_roc.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

