# Subject-level stratified split for the binary MRI dataset.
# Input:
#   Z244_merged_slices_binary.csv
#
# Output:
#   splits_binary/train.csv
#   splits_binary/val.csv
#   splits_binary/test.csv
#
# Format:
#   Subject_ID, Label, rel_path
#   where Label = binary class folder names (Demented, Non_Demented)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

#define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent

CSV_PATH   = SCRIPT_DIR / "Z244_merged_slices_binary.csv"
OUT_DIR    = ROOT / "splits_binary"

#split configuration parameters
SEED       = 42
TEST_RATIO = 0.20
VAL_RATIO  = 0.20


def main():

    print("\n=== SUBJECT-LEVEL STRATIFIED SPLIT (Binary) ===")
    print("CSV:", CSV_PATH)

    # 1) Load the merged slices CSV file
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    assert "Subject_ID" in df.columns, "Missing 'Subject_ID' column"
    assert "Label" in df.columns, "Missing 'Label' column"

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
    print("Distribution:", Counter(labels))

    #split data into training/validation set (80%) and test set (20%) while maintaining class proportions (stratify)
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

    #helper function to write plain text lists of subject IDs
    def write_subject_list(name, sids):
        txt_path = OUT_DIR / f"{name}_subjects.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for sid in sids:
                f.write(f"{sid}\n")
        print(f"{name}_subjects.txt → {len(sids)} subjects")

    #save subject lists for each split
    write_subject_list("train", s_train)
    write_subject_list("val",   s_val)
    write_subject_list("test",  s_test)

    # 5) Build final CSV files containing paths and labels for each split
    def build_split_csv(name, subj_ids):
        rows = []
        for sid in subj_ids:
            #get the string label for the current subject
            lbl = subject_df.loc[subject_df["Subject_ID"] == sid, "Label"].iloc[0]
            #define relative path where the preprocessed .npy file is stored
            rel_path = f"{lbl}/{sid}.npy"
            rows.append([sid, lbl, rel_path])

        #create DataFrame and save to CSV
        out_df = pd.DataFrame(rows, columns=["Subject_ID", "Label", "rel_path"])
        out_path = OUT_DIR / f"{name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"{name}.csv → {len(out_df)} subjects")

    build_split_csv("train", s_train)
    build_split_csv("val",   s_val)
    build_split_csv("test",  s_test)

    # 6) Report and print class distributions for each split
    def report(name, y):
        c = Counter(y)
        print(
            f"{name:5s}: total {len(y)} subjects | " +
            " | ".join(f"{lbl}: {cnt}" for lbl, cnt in sorted(c.items()))
        )

    print("\n=== SUBJECT-LEVEL DISTRIBUTIONS ===")
    report("Train", y_train)
    report("Val",   y_val)
    report("Test",  y_test)

    print("\nDONE – subject-level splits created successfully!\n")


main()
