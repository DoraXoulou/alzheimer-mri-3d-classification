# Subject-level STRATIFIED split for Stage-2 Dementia classifier (Very_mild vs Mild_Moderate).
# Input:
#   Z244_stage2_slices.csv 
#
# Output:
#   splits_stage2_dementia/train.csv
#   splits_stage2_dementia/val.csv
#   splits_stage2_dementia/test.csv
#
# Format:
#   Subject_ID, Label, rel_path
#   where Label = Stage-2 dementia folder names

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


#define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent           
ROOT       = SCRIPT_DIR.parent                        

CSV_PATH   = SCRIPT_DIR / "Z244_stage2_slices.csv"
OUT_DIR    = ROOT / "splits_stage2_dementia"

#split configuration parameters
SEED       = 42
TEST_RATIO = 0.20
VAL_RATIO  = 0.20


def main():
    print(f"[INFO] Loading: {CSV_PATH}")
    
    # 1) Load the merged slices CSV file
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    assert "Subject_ID" in df.columns
    assert "Label" in df.columns

    # 2) Create a subject-level dataframe 
    subject_df = (
        df.groupby("Subject_ID")["Label"]
          .first()
          .reset_index()
    )

    #extract arrays of subject IDs and their corresponding string labels
    subjects = subject_df["Subject_ID"].astype(str).values
    labels   = subject_df["Label"].astype(str).values

    print(f"\nTotal subjects: {len(subjects)}")
    print("Label distribution:", Counter(labels))

    
    #split data into training/validation set (80%) and test set (20%) while maintaining class proportions 
    s_trainval, s_test, y_trainval, y_test = train_test_split(
        subjects, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=SEED
    )

  
    #recalculate validation ratio relative to the remaining trainval set
    effective_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    
    #further split trainval into actual training set and validation set
    s_train, s_val, y_train, y_val = train_test_split(
        s_trainval, y_trainval,
        test_size=effective_val_ratio,
        stratify=y_trainval,
        random_state=SEED
    )

    #ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Helper function to write plain text lists of subject IDs 
    def write_txt(name, sids):
        with open(OUT_DIR / f"{name}_subjects.txt", "w") as f:
            for sid in sids:
                f.write(f"{sid}\n")
        print(f"{name}_subjects.txt → {len(sids)} subjects")

    #save subject lists for each split
    write_txt("train", s_train)
    write_txt("val",   s_val)
    write_txt("test",  s_test)

    # 4) Build final CSV files containing paths and labels for feature extraction 
    def build_csv(name, subj_ids):
        rows = []
        for sid in subj_ids:
            #gt the string label for the current subject
            lbl = subject_df.loc[subject_df["Subject_ID"] == sid, "Label"].iloc[0]
            #define relative path where the preprocessed .npy file is stored
            rel_path = f"{lbl}/{sid}.npy"
            rows.append([sid, lbl, rel_path])

        #create DataFrame and save to CSV
        out_df = pd.DataFrame(rows, columns=["Subject_ID", "Label", "rel_path"])
        out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv → {len(out_df)} subjects")

    build_csv("train", s_train)
    build_csv("val",   s_val)
    build_csv("test",  s_test)

    # 5) Report and print class distributions for each split
    def report(name, y):
        c = Counter(y)
        print(
            f"{name:6}: total {len(y)} | " +
            " | ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(c.items()))
        )

    print("\n=== SUBJECT-LEVEL DISTRIBUTIONS ===")
    report("Train", y_train)
    report("Val",   y_val)
    report("Test",  y_test)

    print("\nDONE ✓  Created Stage-2 dementia subject-level split.\n")


main()
