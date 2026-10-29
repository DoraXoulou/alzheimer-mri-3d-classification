import os
import re
from typing import List
import numpy as np
import csv
from collections import defaultdict

#root directory containing the raw MRI dataset split by class folders
DATA_ROOT = r"C:\Users\theod\alzheimer_detection\data_raw"

#diagnostic class folders to process
CLASS_DIRS = ["Non Demented","Very mild Dementia","Mild Dementia","Moderate Dementia"]

Z = 244 #the number of slices of each subject

#regex to extract the slice index (the last number) from the filename
LAST_NUMBER_RE = re.compile(r"(\d+)(?=\.[^.]+$)") 

#extract and return the slice index from an image filename
def return_last_number(fname: str) -> int:  
    m = LAST_NUMBER_RE.search(fname)
    return int(m.group(1)) if m else 0

#if a brain has D sections
#if D > Z: keep Z central
#if D ≤ Z: keep all 
def pick_Z_centered_slices(D: int, Z: int) -> List[int]: 
    if D <= Z:
        return list(range(D))
    start = (D - Z) // 2
    return list(range(start, start + Z))

#generated uniform-spread indices helper (used only when D < Z)
def spread_indices_uniform(D: int, Z: int) -> List[int]:
    if D <= 1:
        return [0] * Z
    if Z == 1:
        return [round((D - 1) / 2)]
    return [int(round(i * (D - 1) / (Z - 1))) for i in range(Z)]


#select exactly Z slices for each subject
#if there are more: take the center ones
#if there are fewer: apply uniform-spread interpolation 
def select_Z_slices(paths: List[str], Z: int, mode: str = "center") -> List[str]:
    D = len(paths)
    if D == Z:
        return paths
    if D > Z:
        idx = pick_Z_centered_slices(D, Z) 
        return [paths[i] for i in idx]
    #when D < Z we now use uniform-spread indices 
    idx = spread_indices_uniform(D, Z)
    return [paths[i] for i in idx]

#create a csv file with the slices to be used in the CNN
def create_selected_Z_slices_csv(output_csv: str, Z: int = 244, mode: str = "center"):

    col1, col2, col3, col4 = 15, 20, 10, 80  #column widths

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(f"{'Subject_ID':<{col1}} {'Label':<{col2}} {'Z_index':<{col3}} {'Filepath':<{col4}}\n")
        f.write("-" * (col1 + col2 + col3 + col4 + 3) + "\n")
        
        #iterate through each diagnostic class folder
        for class_name in CLASS_DIRS:
            class_dir = os.path.join(DATA_ROOT, class_name)
            if not os.path.isdir(class_dir):
                print(f"Missing class dir: {class_dir}")
                continue
            
            #group all image file paths by subject ID 
            subject_slices = defaultdict(list)
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(".jpg"):
                    continue

                #filename: OAS1_0001_MR1_mpr-1_101.jpg
                #extract the subject_id part: "OAS1_0001_MR1"
                sid = "_".join(fname.split("_")[:3])
                fp = os.path.join(class_dir, fname)
                subject_slices[sid].append(fp)
            
            
            num_subjects = 0
            for sid, paths in subject_slices.items():
                #sort slices numerically based on their extracted index to preserve anatomical order
                paths.sort(key=lambda p: return_last_number(os.path.basename(p)))

                #select exactly Z slices (centered cropping or uniform distribution)
                fixed_paths = select_Z_slices(paths, Z=Z, mode=mode)

                #write each selected slice as a separate line in the CSV
                for z, p in enumerate(fixed_paths):
                    f.write(f"{sid:<{col1}} {class_name:<{col2}} {z:<{col3}} {p:<{col4}}\n")
                num_subjects += 1

            print(f"{class_name}: wrote {num_subjects} subjects @ Z={Z} ({mode})")

    print(f"\nReady: {output_csv}")

def main():
    mode = "center"
    output_csv = f"Z{Z}_centered_slices.csv"
    create_selected_Z_slices_csv(output_csv=output_csv, Z=Z, mode=mode)
    print(f"\nThe {output_csv} is ready")

main()

