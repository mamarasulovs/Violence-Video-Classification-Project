import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import ViolenceVideoDataset, VideoConfig
from model import build_model


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path("artifacts") / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("Missing artifacts/best.pt. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = VideoConfig(**ckpt["video_cfg"])

    ds = ViolenceVideoDataset("splits.csv", "val", cfg, seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.argmax(logits, dim=1).cpu()
        y_true.extend(y.tolist())
        y_pred.extend(p.tolist())

    acc = float(accuracy_score(y_true, y_pred))
    rep_txt = classification_report(y_true, y_pred, target_names=["NonViolence", "Violence"])
    rep_dict = classification_report(y_true, y_pred, target_names=["NonViolence", "Violence"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("Validation accuracy:", acc)
    print(rep_txt)
    print("Confusion matrix:\n", cm)

    Path("results").mkdir(exist_ok=True)
    out = {
        "accuracy": acc,
        "classification_report": rep_dict,
        "confusion_matrix": cm.tolist(),
    }
    (Path("results") / "val_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved results/val_report.json")


if __name__ == "__main__":
    main()
