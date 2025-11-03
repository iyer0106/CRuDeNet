import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from typing import Optional, Tuple, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def normalize01(arr: np.ndarray) -> np.ndarray:
    amin, amax = float(arr.min()), float(arr.max())
    if amax == amin:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


class VolumeDataset(Dataset):
    """Yields temporal patches [1, T, H, W] from raw and gt stacks.

    raw and gt can be file paths to .tif or numpy arrays.
    """

    def __init__(
        self,
        raw: Union[str, np.ndarray],
        gt: Union[str, np.ndarray],
        patch_t: int,
        patch_xy: int,
        normalize: bool = True,
    ) -> None:
        if isinstance(raw, str):
            raw_stack = tifffile.imread(raw).astype(np.float32)
        else:
            raw_stack = raw.astype(np.float32)

        if isinstance(gt, str):
            gt_stack = tifffile.imread(gt).astype(np.float32)
        else:
            gt_stack = gt.astype(np.float32)

        if normalize:
            raw_stack = normalize01(raw_stack)
            gt_stack = normalize01(gt_stack)

        self.raw = raw_stack
        self.gt = gt_stack
        self.T = int(patch_t)
        self.XY = int(patch_xy)
        self.h, self.w = raw_stack.shape[1:]
        self.num = raw_stack.shape[0] - self.T

    def __len__(self) -> int:
        return max(0, self.num)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_seq = self.raw[idx : idx + self.T]
        gt_seq = self.gt[idx : idx + self.T]

        x0 = np.random.randint(0, self.h - self.XY + 1)
        y0 = np.random.randint(0, self.w - self.XY + 1)
        raw_seq = raw_seq[:, x0 : x0 + self.XY, y0 : y0 + self.XY]
        gt_seq = gt_seq[:, x0 : x0 + self.XY, y0 : y0 + self.XY]

        x = torch.from_numpy(raw_seq[None]).float()  # [1,T,H,W]
        y = torch.from_numpy(gt_seq[None]).float()   # [1,T,H,W]
        return x, y


