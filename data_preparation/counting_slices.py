import os
import re
from typing import List
from collections import defaultdict
from statistics import mean, median
import csv

DATA_ROOT = r"C:\Users\theod\alzheimer_detection\data_raw"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV    = os.path.join(SCRIPT_DIR, "slices_per_subject.csv")

CLASS_DIRS: List[str] = ["Non Demented","Very mild Dementia","Mild Dementia","Moderate Dementia",] #classes

IMG_EXTS = {".jpg", ".jpeg"} #acceptable image formats


#function to check if the file has an image extension
def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS

#REGEX (REgular EXpression) to extract the subject_id from the filename
SUBJECT_RE = re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE)  

#function that takes a slice filename and returns the unique subject ID
def extract_subject_id(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] #remove file extension
    m = SUBJECT_RE.search(base) #search for the subject id pattern 
    return m.group(1) if m else base #return id if found, otherwise the full name

#function that counts how many slices each subject has within a class folder
def count_slice_in_subject(class_dir: str):
    slices_per_subject = defaultdict(int)   #dict: {subject_id: number_of_slices}
    total_images = 0 #counter for total images in the folder

    for fname in os.listdir(class_dir):
        if not is_image(fname):
            continue
           
        total_images += 1 #if it's an image, increase the total image counter of the folder
        sid = extract_subject_id(fname) #extract the subject_id (sid) for this specific slice   
        slices_per_subject[sid] +=1     #increase the slice counter for this specific subject

    return slices_per_subject, total_images


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"[ERROR] There isn't DATA ROOT: {DATA_ROOT}")
        return

    #column widths for aligned text table in CSV
    col1_w, col2_w, col3_w = 20, 15, 10

    #open CSV file for writing
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(f"{'Class':<{col1_w}} | {'Subject ID':<{col2_w}} | {'Slices':<{col3_w}}\n")  # header

        #loop through each class
        for class_name in CLASS_DIRS:
            class_dir = os.path.join(DATA_ROOT, class_name)
            if not os.path.isdir(class_dir):
                print(f"Class folder not found: {class_dir}")
                continue

            print(f"\nCounting slices per subject in class: {class_name}")
            sps, total_imgs = count_slice_in_subject(class_dir) #call counting function 

            if not sps:
                print("There are not images")
                continue

            subjects = list(sps.keys()) #list of subject ids
            counts = list(sps.values()) #list of their slice counts

            print(f"Total subjects: {len(subjects)}")
            print(f"Total brain scans: {total_imgs}")
            print(f"slices/subject: min={min(counts)}, max={max(counts)}, median={median(counts)}")

            for sid, n in sps.items():
                f.write(f"{class_name:<{col1_w}} | {sid:<{col2_w}} | {n:<{col3_w}}\n")

    print(f"\nSaved CSV: {OUT_CSV}")



main()
