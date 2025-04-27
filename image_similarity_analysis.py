#!/usr/bin/env python3
"""
TNM098 – Advanced Visual Data Analysis
Lab 3.1 helper: image‑similarity analysis pipeline (folder version)
-----------------------------------------------------------------
This script automates the workflow for Lab 3.1 when the 12 images are
already **extracted into a directory** named (by default) `Lab3.1`.

Workflow
========
1.  **Load** all image files in the specified folder (default `Lab3.1`).
2.  **Compute** a multi‑faceted feature vector for each image (≈ 354 dims):
      * Global colour histogram (RGB, 8 bins/channel)
      * Central‑patch colour histogram
      * Four radial ring colour histograms
      * Luminance histogram
      * Edge‑orientation histogram
      * Edge‑density grid
      * Hu image moments
3.  **Distance matrix**: build 12 × 12 pairwise distances (Euclidean by default).
4.  **Rank** the remaining 11 images by similarity to a user‑chosen anchor.
5.  **(Optional) Visualise** the matrix and similarity gallery.

Usage
-----
```bash
python image_similarity_analysis.py --indir Lab3.1 --anchor 0 --metric euclidean --show
```
Arguments
~~~~~~~~~
* `--indir` : directory holding the 12 images (default: `Lab3.1`)
* `--anchor`: index (0‑based) of the reference image (default: 0)
* `--metric`: distance metric (any supported by `scipy.spatial.distance`)
* `--show`  : display heat‑map and ranked gallery

Dependencies
------------
```bash
pip install opencv-python scikit-image scipy pillow matplotlib pandas tqdm
```

Author: ChatGPT (o3)
-----------------------------------------------------------------
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import distance as dist
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utility: image discovery in a folder
# ---------------------------------------------------------------------------

def discover_images(dir_path: Path) -> List[Path]:
    """Return a sorted list of image paths inside *dir_path*."""
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in exts)
    if len(images) != 12:
        print(f"[!] Expected 12 images, found {len(images)} – continuing anyway…")
    return images

# ---------------------------------------------------------------------------
# Feature extraction helpers (same as before)
# ---------------------------------------------------------------------------

_HIST_BINS = 8  # per channel for colour histograms
_RING_BINS = 4  # per channel, per ring
_PATCH_RATIO = 0.33  # central patch size (fraction of min(H,W))


def _rgb_hist(img: np.ndarray, bins: int = _HIST_BINS) -> np.ndarray:
    chans = cv2.split(img)
    hist = [cv2.calcHist([c], [0], None, [bins], [0, 256]) for c in chans]
    h = np.concatenate(hist).astype('float32').flatten()
    h /= h.sum() + 1e-7
    return h


def _central_patch(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = int(min(h, w) * _PATCH_RATIO)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    patch = img[y0:y0 + size, x0:x0 + size]
    return _rgb_hist(patch)


def _ring_hist(img: np.ndarray, rings: int = 4, bins: int = _RING_BINS) -> np.ndarray:
    h, w = img.shape[:2]
    centre = np.array([w / 2, h / 2])
    yy, xx = np.indices((h, w))
    rr = np.linalg.norm(np.stack([xx - centre[0], yy - centre[1]], axis=2), axis=2)
    r_max = rr.max()
    feats = []
    for i in range(rings):
        mask = (rr >= (i / rings) * r_max) & (rr < ((i + 1) / rings) * r_max)
        masked = cv2.bitwise_and(img, img, mask=mask.astype('uint8'))
        feats.append(_rgb_hist(masked, bins))
    return np.concatenate(feats)


def _luminance_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Y = lab[..., 0]
    hist = cv2.calcHist([Y], [0], None, [bins], [0, 256]).flatten().astype('float32')
    hist /= hist.sum() + 1e-7
    return hist


def _edge_features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Orientation histogram
    bins = 8
    hist, _ = np.histogram(ang, bins=bins, range=(0, 360), weights=mag)
    hist = hist.astype('float32')
    hist /= hist.sum() + 1e-7
    # Edge density grid 6×6
    grid = 6
    h, w = gray.shape
    ph, pw = h // grid, w // grid
    dens = [mag[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].mean() for r in range(grid) for c in range(grid)]
    dens = np.array(dens, dtype='float32')
    if dens.sum() > 0:
        dens /= dens.sum()
    return hist, dens


def _hu_moments(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten().astype('float32')
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-7)
    return hu


def compute_feature_vector(img: np.ndarray) -> np.ndarray:
    feats = [
        _rgb_hist(img),
        _central_patch(img),
        _ring_hist(img),
        _luminance_hist(img),
    ]
    edge_hist, edge_dens = _edge_features(img)
    feats.extend([edge_hist, edge_dens, _hu_moments(img)])
    vec = np.concatenate(feats)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec

# ---------------------------------------------------------------------------
# Matrix computation and ranking (unchanged)
# ---------------------------------------------------------------------------

def build_feature_matrix(image_paths: List[Path]) -> np.ndarray:
    vectors = []
    for p in tqdm(image_paths, desc='Computing features'):
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        vectors.append(compute_feature_vector(img))
    return np.vstack(vectors)


def distance_matrix(features: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    return dist.cdist(features, features, metric=metric)


def rank_similar(dmat: np.ndarray, anchor: int):
    order = np.argsort(dmat[anchor])
    return [i for i in order if i != anchor]

# ---------------------------------------------------------------------------
# Visualisation helper (unchanged)
# ---------------------------------------------------------------------------

def show_results(dmat, ranks, image_paths, anchor):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    # Heat‑map
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(dmat, cmap='viridis')
    ax0.set_title('Distance matrix')
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_xticks(range(len(image_paths)))
    ax0.set_yticks(range(len(image_paths)))
    # Similarity gallery
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    cols = 3
    rows = int(np.ceil((len(ranks) + 1) / cols))
    gs2 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[1], wspace=0.05, hspace=0.2)
    # Anchor
    anchor_img = cv2.cvtColor(cv2.imread(str(image_paths[anchor])), cv2.COLOR_BGR2RGB)
    axA = fig.add_subplot(gs2[0])
    axA.imshow(anchor_img)
    axA.set_title(f'Anchor #{anchor}')
    axA.axis('off')
    # Ranked
    for k, idx in enumerate(ranks):
        r = (k + 1) // cols
        c = (k + 1) % cols
        ax = fig.add_subplot(gs2[r, c])
        img_rgb = cv2.cvtColor(cv2.imread(str(image_paths[idx])), cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"#{idx}\nd={dmat[anchor, idx]:.3f}")
        ax.axis('off')
    fig.suptitle('Similarity ranking')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Image similarity analysis (folder version) for TNM098 Lab 3.1.')
    parser.add_argument('--indir', type=Path, default=Path('Lab3.1'), help='Directory with the 12 images')
    parser.add_argument('--anchor', type=int, default=0, help='Index of anchor image (0‑based)')
    parser.add_argument('--metric', type=str, default='euclidean', help='Distance metric (scipy.spatial)')
    parser.add_argument('--show', action='store_true', help='Display visualisations')
    args = parser.parse_args()

    # Discover images
    images = discover_images(args.indir)
    print(f"[+] Found {len(images)} image(s) in {args.indir}")

    if len(images) < 2:
        raise RuntimeError('Need at least 2 images for similarity analysis.')
    if not (0 <= args.anchor < len(images)):
        raise ValueError('Anchor index out of range.')

    # Features
    feats = build_feature_matrix(images)
    print(f"[+] Feature matrix shape: {feats.shape}")

    # Distance matrix
    dmat = distance_matrix(feats, args.metric)
    pd.DataFrame(dmat).to_csv('distance_matrix.csv', float_format='%.6f', index=False, header=False)
    print('[+] Saved distance_matrix.csv')

    # Ranking
    ranks = rank_similar(dmat, args.anchor)
    print(f"[+] Images most similar to {args.anchor}: {ranks}")

    # Visualise
    if args.show:
        show_results(dmat, ranks, images, args.anchor)

if __name__ == '__main__':
    main()
