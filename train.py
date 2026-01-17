import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score

from dataset import ViolenceVideoDataset, VideoConfig
from model import build_model

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    ys, preds = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        p = torch.argmax(logits, dim=1)
        ys.extend(y.cpu().tolist())
        preds.extend(p.cpu().tolist())
    return float(accuracy_score(ys, preds))


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    #cfg = VideoConfig(num_frames=32, resize_short=256, crop_size=224)
    cfg = VideoConfig(num_frames=32, resize_short=256, crop_size=224)

    train_ds = ViolenceVideoDataset("splits.csv", "train", cfg, seed=42)
    val_ds = ViolenceVideoDataset("splits.csv", "val", cfg, seed=42)

    # ===== IMPORTANT: Colab stability settings =====
    # Start with num_workers=0 (most stable).
    # If everything is stable and you want speed, try num_workers=2 later.
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )
    # ==============================================

    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

    # New AMP API (optional, removes warnings)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_ckpt_path = Path("artifacts") / "best.pt"

    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / max(1, len(pbar)))

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "video_cfg": cfg.__dict__,
                    "best_acc": best_acc,
                },
                best_ckpt_path,
            )
            print(f"Saved best -> {best_ckpt_path} (acc={best_acc:.4f})")

    print("Best val accuracy:", best_acc)

    # Export to safetensors (required by task)
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"]

    out_st = Path("artifacts") / "model.safetensors"
    save_file(state_dict, str(out_st))
    print("Saved:", out_st)

    # Save config + labels mapping
    config = {
        "arch": "r2plus1d_18",
        "num_classes": 2,
        "video_cfg": ckpt["video_cfg"],
        "best_val_acc": ckpt["best_acc"],
    }
    (Path("artifacts") / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    labels = {"0": "NonViolence", "1": "Violence"}
    (Path("artifacts") / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
