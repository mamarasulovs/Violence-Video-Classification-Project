import csv
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Optional decord (fast). If unavailable, fallback to OpenCV.
try:
    import decord
    decord.bridge.set_bridge("native")
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


@dataclass
class VideoConfig:
    num_frames: int = 16
    resize_short: int = 256
    crop_size: int = 224
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)


def get_num_frames(path: str) -> int:
    """Get total number of frames without decoding everything."""
    if HAS_DECORD:
        try:
            vr = decord.VideoReader(path)
            return len(vr)
        except Exception:
            pass

    if not HAS_CV2:
        return 0

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frames_by_indices(path: str, indices: List[int]) -> np.ndarray:
    """
    Read only selected frames.
    Returns (T,H,W,3) in RGB.
    """
    if HAS_DECORD:
        vr = decord.VideoReader(path)
        n = len(vr)
        # clamp indices
        indices = [min(max(int(i), 0), n - 1) for i in indices]
        frames = vr.get_batch(indices).asnumpy()  # (T,H,W,3) RGB
        return frames

    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is not installed and decord is unavailable.")

    cap = cv2.VideoCapture(path)
    frames = []
    last = None

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()

        if not ok or frame is None:
            # If decode fails, repeat last valid frame to keep shape consistent
            if last is not None:
                frames.append(last)
                continue
            # If nothing valid yet, try reading the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Could not decode video: {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last = frame
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)


def sample_indices(num_total: int, num_frames: int, train: bool, rng: random.Random) -> List[int]:
    if num_total <= 0:
        return [0] * num_frames

    if num_total < num_frames:
        idx = list(range(num_total))
        while len(idx) < num_frames:
            idx += idx
        return idx[:num_frames]

    base = np.linspace(0, num_total - 1, num_frames).astype(int)

    # small temporal jitter during training
    if train:
        jitter = rng.randint(-2, 2)
        base = np.clip(base + jitter, 0, num_total - 1)

    return base.tolist()


def resize_short_side(frames: torch.Tensor, short: int) -> torch.Tensor:
    # frames: (T,C,H,W)
    T, C, H, W = frames.shape
    if H < W:
        new_h = short
        new_w = int(round(W * (short / H)))
    else:
        new_w = short
        new_h = int(round(H * (short / W)))

    out = []
    for t in range(T):
        out.append(TF.resize(frames[t], [new_h, new_w], antialias=True))
    return torch.stack(out, dim=0)


def center_crop(frames: torch.Tensor, size: int) -> torch.Tensor:
    out = []
    for t in range(frames.shape[0]):
        out.append(TF.center_crop(frames[t], [size, size]))
    return torch.stack(out, dim=0)


def random_resized_crop(frames: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    # Same crop for all frames
    T, C, H, W = frames.shape
    area = H * W

    for _ in range(10):
        target_area = area * rng.uniform(0.6, 1.0)
        aspect = rng.uniform(0.75, 1.333)

        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))

        if 0 < crop_h <= H and 0 < crop_w <= W:
            i = rng.randint(0, H - crop_h)
            j = rng.randint(0, W - crop_w)
            out = []
            for t in range(T):
                out.append(TF.resized_crop(frames[t], i, j, crop_h, crop_w, [size, size], antialias=True))
            return torch.stack(out, dim=0)

    return center_crop(frames, size)


class ViolenceVideoDataset(Dataset):
    def __init__(self, splits_csv: str, split: str, cfg: VideoConfig, seed: int = 42):
        self.cfg = cfg
        self.split = split
        self.seed = seed
        self.items: List[Tuple[str, int]] = []

        with open(splits_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["split"] == split:
                    self.items.append((row["path"], int(row["label"])) )

        if not self.items:
            raise RuntimeError(f"No items for split='{split}' in {splits_csv}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        rng = random.Random(self.seed + idx)

        # Try a few times in case of corrupted decode
        for attempt in range(3):
            try:
                n = get_num_frames(path)
                inds = sample_indices(
                    n, self.cfg.num_frames, train=(self.split == "train"), rng=rng
                )
                clip_np = read_frames_by_indices(path, inds)  # (T,H,W,3) RGB

                # (T,3,H,W) float in [0,1]
                clip = torch.from_numpy(clip_np).to(torch.uint8)
                clip = clip.permute(0, 3, 1, 2).contiguous()
                clip = clip.float() / 255.0

                clip = resize_short_side(clip, self.cfg.resize_short)

                if self.split == "train":
                    clip = random_resized_crop(clip, self.cfg.crop_size, rng)
                    if rng.random() < 0.5:
                        clip = torch.flip(clip, dims=[3])
                else:
                    clip = center_crop(clip, self.cfg.crop_size)

                mean = torch.tensor(self.cfg.mean).view(1, 3, 1, 1)
                std = torch.tensor(self.cfg.std).view(1, 3, 1, 1)
                clip = (clip - mean) / std

                # Return (C,T,H,W)
                clip = clip.permute(1, 0, 2, 3).contiguous()

                return clip, torch.tensor(label, dtype=torch.long), path

            except Exception:
                # On failure, tweak RNG and retry
                rng = random.Random(self.seed + idx + 1000 * (attempt + 1))

        # If it still fails after retries, return a safe "blank" clip to avoid crashing
        blank = torch.zeros((3, self.cfg.num_frames, self.cfg.crop_size, self.cfg.crop_size), dtype=torch.float32)
        return blank, torch.tensor(label, dtype=torch.long), path
