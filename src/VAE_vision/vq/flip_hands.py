"""
Reflect hands_augmented.npy across the y-axis to produce mirrored right-hand data.
Input:  data/hands_augmented.npy  (N, 128, 128, 3) uint8
Output: data/hands_right.npy      (N, 128, 128, 3) uint8
"""
import numpy as np
from pathlib import Path

src = Path("data/hands_augmented.npy")
dst = Path("data/hands_right.npy")

print(f"Loading {src} ...")
arr = np.load(src)
print(f"  shape: {arr.shape}  dtype: {arr.dtype}")

flipped = arr[:, :, ::-1, :].copy()  # flip along W axis, copy to drop negative stride

print(f"Saving {dst} ...")
np.save(dst, flipped)
print(f"Done. {dst}  {flipped.shape}  {flipped.dtype}")
