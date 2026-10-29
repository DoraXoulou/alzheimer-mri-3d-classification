import os
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.ndimage import (
    gaussian_filter, binary_closing, binary_opening,
    label, generate_binary_structure
)


MODE = "binary"

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

#set input source and output directories for preprocessed volumes and masks
if MODE == "binary":
    source_dir = PROJECT_ROOT / "dataset_npy_binary"
    destination_subject_dir = PROJECT_ROOT / "dataset_preprocessed_binary"
    destination_mask_dir    = PROJECT_ROOT / "dataset_preprocessed_masks_binary"
else:
    raise ValueError("Only binary mode supported here")

#skull stripping - brain extraction
mask_fill_radius = 2
mask_clean_radius = 1
min_brain_component_ratio = 0.15

#bias field correction - brightness
bias_sigma = 20.0
bias_eps   = 1e-6

#noise smoothing
smooth_sigma = 0.6

#clipping
clip_lo, clip_hi = -5.0, 5.0


#create a directory if it does not already exist
def create_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

#find an automatic threshold to separate the brain from the background
def otsu_thresholding(subject: np.ndarray, nbins: int = 256) -> float:

    #keep only the valid values (discard any NaN or Inf)  
    voxels = subject[np.isfinite(subject)]

    #if the array is empty, simply return the mean value
    if voxels.size == 0:
        return float(np.mean(subject))

    #find the extreme bounds to avoid outliers
    lo, hi = np.percentile(voxels, [1, 99])

    #clip/restrict the values within these bounds
    voxels = np.clip(voxels, lo, hi)

    #create a histogram of the intensities dividen into 256 bins
    hist, bins = np.histogram(voxels, bins=nbins, range=(lo, hi))

    #calculate the probability of each intensity by dividing by the total sum
    prob = hist.astype(float) / max(hist.sum(), 1)

    #calculate cumulative probabilities (omega)
    omega = np.cumsum(prob)

    #calculate cumulative mean values (mu) for each bin
    mu = np.cumsum(prob * (bins[:-1] + bins[1:]) * 0.5)

    #get the overall mean value (the last element)
    mu_t = mu[-1]

    #calculate the between-class variance (Otsu's formula)
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-12)

    #find the index where variance is maximum (the optimal threshold point)
    idx = np.nanargmax(sigma_b2)

    #calculate and return the final threshold value between the corresponding bins
    return float((bins[idx] + bins[idx + 1]) * 0.5)

#generate a spherical shape (ball) for morphological operations on the image
def ball_struct(radius: int):
    if radius <= 0:
        return np.ones((1,1,1), dtype=bool)
    r = radius
    zz,yy,xx = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    return (zz**2 + yy**2 + xx**2) <= r*r


#keep only the largest connected region (the brain) and discards small random noise pieces
def largest_cc(mask: np.ndarray, min_ratio: float = 0.1):
    if mask.sum() == 0:
        return mask

    conn = generate_binary_structure(3, 2)
    labs, n = label(mask, structure=conn)

    if n <= 1:
        return mask

    sizes = np.bincount(labs.ravel())
    sizes[0] = 0
    k = sizes.argmax()

    largest = sizes[k]
    keep = (sizes >= largest * min_ratio)
    keep[0] = False

    return np.isin(labs, np.where(keep)[0])

#remove the skull and background, leaving only the clean brain (create the mask)
def skull_stripping(volume: np.ndarray) -> np.ndarray:
    thresh = otsu_thresholding(volume)
    mask = volume > thresh

    mask = binary_closing(mask, structure=ball_struct(mask_fill_radius))
    mask = binary_opening(mask, structure=ball_struct(mask_clean_radius))

    mask = largest_cc(mask, min_ratio=min_brain_component_ratio)
    return mask.astype(bool)

#correct brightness/shading inconsistencies caused by the scanner
def bias_field_correction(subject: np.ndarray, mask: np.ndarray):
    subject = subject.astype(np.float32)
    filled = subject.copy()

    if mask.any():
        med = np.median(filled[mask])
        filled[~mask] = med
    else:
        med = np.median(filled)

    bias = gaussian_filter(filled, sigma=bias_sigma)
    bias = np.clip(bias, bias_eps, None)

    corrected = subject / bias

    if mask.any():
        scale = np.median(corrected[mask])
        if abs(scale) > 1e-6:
            corrected = corrected / scale

    return corrected.astype(np.float32)


#bring brain values to a common scale (mean=0, std=1) exclusively inside the mask
def zscore_in_mask(volume: np.ndarray, mask: np.ndarray):
    if not mask.any():
        m, s = volume.mean(), volume.std()
        return (volume - m) / (s if s > 1e-6 else 1)

    inside = volume[mask]
    m, s = inside.mean(), inside.std()

    out = (volume - m) / (s if s > 1e-6 else 1)
    out[~mask] = 0
    return out

#execute the complete preprocessing pipeline step by step for each volume
def preprocess(vol: np.ndarray):
    mask = skull_stripping(vol)

    vol = bias_field_correction(vol, mask)
    vol = gaussian_filter(vol, sigma=smooth_sigma)
    vol = np.clip(vol, clip_lo, clip_hi)
    vol = zscore_in_mask(vol, mask)

    return vol.astype(np.float32), mask.astype(np.uint8)


def main():

    create_dir(destination_subject_dir)
    create_dir(destination_mask_dir)

    print(f"\n[INFO] Preprocessing binary dataset")
    print(f"[INFO] Source: {source_dir}")

    #loop through each class folder 
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name  # "Demented" or "Non_Demented"
        print(f"\n[INFO] Class: {label}")

        out_vol_dir = destination_subject_dir / label
        out_msk_dir = destination_mask_dir / label
        create_dir(out_vol_dir)
        create_dir(out_msk_dir)

        #get all the .npy files for the current class
        npy_files = sorted(class_dir.glob("*.npy"))
        print(f"[INFO] Files: {len(npy_files)}")

        #process each subject file individually
        for i, path in enumerate(npy_files, 1):
            vol = np.load(path)
            pre, msk = preprocess(vol)

            out_vol = out_vol_dir / path.name
            out_msk = out_msk_dir / (path.stem + "_mask.npy")

            #save the cleaned volume and its corresponding mask
            np.save(out_vol, pre)
            np.save(out_msk, msk)

            if i % 50 == 0 or i == len(npy_files):
                print(f"  - done {i}/{len(npy_files)}")

    print("\n[INFO] Preprocessing COMPLETE ✓")
    print(f"[INFO] Saved preprocessed volumes at: {destination_subject_dir}")
    print(f"[INFO] Saved brain masks at:        {destination_mask_dir}")


main() 
