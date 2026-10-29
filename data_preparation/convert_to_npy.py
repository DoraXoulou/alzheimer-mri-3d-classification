import os
import cv2
import numpy as np
from scipy.ndimage import zoom  
import pandas as pd  

#choose the diagnostic scenario strategy
MODE = "binary"   #binary, dementia, 3class

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


if MODE == "3class":
    CSV_PATH   = os.path.join(SCRIPT_DIR, "Z244_merged_slices_3class.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_npy_3classes")
elif MODE == "binary":
    CSV_PATH   = os.path.join(SCRIPT_DIR, "Z244_merged_slices_binary.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_npy_binary")
elif MODE == "dementia":
    CSV_PATH   = os.path.join(SCRIPT_DIR, "Z244_merged_slices_dementia.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_npy_dementia")
else:
    raise ValueError(f"Unknown MODE='{MODE}")

#target 3D volume dimensions for CNN input (Depth, Height, Width)
TARGET_SHAPE = (160, 160, 160)  
Z = 244  #number of slices per subject

#ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv(csv_path):
    records = []

    #load csv with pandas
    df = pd.read_csv(csv_path)

    
    cols = {c.strip(): c for c in df.columns} 
    try:
        col_sid   = cols["Subject_ID"]
        col_label = cols["Label"]
        col_z     = cols["Z_index"]
        col_path  = cols["Filepath"]
    except KeyError:
        raise ValueError(f"Expected columns not found in CSV. Found: {list(df.columns)}")

    
    for _, row in df.iterrows():
        sid      = str(row[col_sid]).strip()
        label    = str(row[col_label]).strip()
        filepath = str(row[col_path]).strip()

        try:
            z_idx = int(row[col_z])
        except Exception:
            continue

        records.append((sid, label, z_idx, filepath))

    return records

#read a single 2D MRI slice image in grayscale and casts it to float32
def convert_to_npy(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #read image
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = img.astype(np.float32) #convert to float32

    return img


def main():
    print(f"Loading slice from {CSV_PATH}")
    data = load_csv(CSV_PATH)
    print(f"Loaded {len(data)} slices total")

    #group individual slices by subject ID and diagnostic label
    grouped = {}
    for sid, label, z_idx, filepath in data:
        grouped.setdefault((sid, label), []).append((z_idx, filepath))

    print(f"\nFound {len(grouped)} subjects")

    Z_tgt, H_tgt, W_tgt = TARGET_SHAPE 

    for (sid, label), slices in grouped.items():
        #sort slices by z-index
        slices.sort(key=lambda x: x[0])

        if len(slices) != Z:
            print(f"[WARN] Subject {sid} has {len(slices)} slices (expected {Z})")

        #read all slices for the current subject
        imgs = []
        for z, fp in slices:
            img = convert_to_npy(fp)
            imgs.append(img)

        #convert to numpy array (Z, H, W)
        #create a 3D volume (Z×H×W)
        #each subject as a volume for CNN
        #stack slices to construct the 3D numpy array volume of shape
        subject_volume = np.stack(imgs, axis=0)

        Z_raw, H_raw, W_raw = subject_volume.shape
        #calculate spatial scale factors to match target dimensions
        sz = Z_tgt / max(Z_raw, 1)
        sy = H_tgt / max(H_raw, 1)
        sx = W_tgt / max(W_raw, 1)
        
        subject_volume = zoom(subject_volume, zoom=(sz, sy, sx), order=1)  

        #apply Z-score normalization for stable CNN training
        mean = subject_volume.mean()
        std = subject_volume.std()
        if std < 1e-6:
            std = 1.0
        subject_volume = (subject_volume - mean) / std

        #create output filename
        label_dir = os.path.join(OUTPUT_DIR, label.replace(" ", "_"))
        os.makedirs(label_dir, exist_ok=True)
        out_path = os.path.join(label_dir, f"{sid}.npy")

        #save as .npy
        np.save(out_path, subject_volume)

    print("\nMRI volumes saved in:", OUTPUT_DIR)


main()
