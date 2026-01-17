import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

LABELS = {
    "NonViolence": 0,
    "Violence": 1,
}

def collect_videos(dataset_root: Path):
    items = []
    for cls_name, label in LABELS.items():
        cls_dir = dataset_root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                items.append((str(p), label))
    if not items:
        raise RuntimeError(f"No videos found under: {dataset_root}")
    return items

def main():
    dataset_root = Path("data") / "Real Life Violence Dataset"
    items = collect_videos(dataset_root)

    paths = [p for p, _ in items]
    labels = [y for _, y in items]

    train_p, val_p, train_y, val_y = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    out = Path("splits.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for p, y in zip(train_p, train_y):
            w.writerow([p, y, "train"])
        for p, y in zip(val_p, val_y):
            w.writerow([p, y, "val"])

    print(f"Saved {out}  train={len(train_p)}  val={len(val_p)}")

if __name__ == "__main__":
    main()
