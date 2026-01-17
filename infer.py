import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import (
    VideoConfig,
    get_num_frames,
    read_frames_by_indices,
    resize_short_side,
    center_crop,
)
from model import build_model


def preprocess_video(path: str, cfg: VideoConfig) -> torch.Tensor:
    """
    Load a video and build a (C,T,H,W) tensor normalized for the model.
    Uses the same uniform sampling strategy as validation (no random aug).
    """
    n = get_num_frames(path)
    T = cfg.num_frames

    if n <= 0:
        # fallback: return blank
        clip = torch.zeros((3, T, cfg.crop_size, cfg.crop_size), dtype=torch.float32)
        return clip

    if n < T:
        idx = list(range(n))
        while len(idx) < T:
            idx += idx
        idx = idx[:T]
    else:
        idx = torch.linspace(0, n - 1, T).long().tolist()

    # Read only sampled frames (T,H,W,3) RGB
    clip_np = read_frames_by_indices(path, idx)

    # (T,3,H,W) float in [0,1]
    clip = torch.from_numpy(clip_np).to(torch.uint8)
    clip = clip.permute(0, 3, 1, 2).contiguous().float() / 255.0

    # Resize + center crop (no random transforms for inference)
    clip = resize_short_side(clip, cfg.resize_short)
    clip = center_crop(clip, cfg.crop_size)

    # Normalize
    mean = torch.tensor(cfg.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.std).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    # (C,T,H,W)
    clip = clip.permute(1, 0, 2, 3).contiguous()
    return clip


@torch.no_grad()
def predict(model, x: torch.Tensor, device: str):
    model.eval()
    x = x.unsqueeze(0).to(device, non_blocking=True)  # (1,C,T,H,W)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="Path to a single video file")
    ap.add_argument("--folder", type=str, default=None, help="Folder containing videos")
    ap.add_argument("--out", type=str, default="results/external_predictions.json")
    args = ap.parse_args()

    ckpt_path = Path("artifacts") / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("Missing artifacts/best.pt. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = VideoConfig(**ckpt["video_cfg"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    labels = {0: "NonViolence", 1: "Violence"}

    videos = []
    if args.video:
        videos.append(Path(args.video))
    if args.folder:
        p = Path(args.folder)
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            videos.extend(p.glob(ext))

    if not videos:
        raise SystemExit("No videos found. Provide --video or --folder, and ensure files are mp4/avi/mov/mkv.")

    results = []
    for vp in videos:
        x = preprocess_video(str(vp), cfg)
        pred, probs = predict(model, x, device)
        rec = {
            "video": str(vp),
            "pred_label": labels[pred],
            "prob_nonviolence": float(probs[0]),
            "prob_violence": float(probs[1]),
        }
        results.append(rec)
        print(f"{vp.name}: {labels[pred]} (p_violence={probs[1]:.3f})")

    Path("results").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
