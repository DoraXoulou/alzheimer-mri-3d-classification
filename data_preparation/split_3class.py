# Subject-level stratified split for the 3-class MRI dataset.
# Input:
#   Z244_merged_slices_3class.csv
#
# Output:
#   splits_3class/train.csv
#   splits_3class/val.csv
#   splits_3class/test.csv
#
# Format:
#   Subject_ID, Label, rel_path
#   where Label = {0: Non_Demented, 1: Very_mild_Dementia, 2: Mild_Moderate_Dementia}


from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

#define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent

CSV_PATH   = SCRIPT_DIR / "Z244_merged_slices_3class.csv"
OUT_DIR    = ROOT / "splits_3class"

#split configuration parameters
SEED       = 42
TEST_RATIO = 0.20
VAL_RATIO  = 0.20

#label mapping dictionary for the 3 classes 
LABEL_MAP = {
    "Non_Demented": 0,
    "Very_mild_Dementia": 1,
    "Mild_Moderate_Dementia": 2,
}


def main():

    print("\n=== SUBJECT-LEVEL STRATIFIED SPLIT (3-class) ===")
    print("CSV:", CSV_PATH)

    # 1) Load the merged slices CSV file
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    # 2) Create a subject - level dataframe 
    subject_df = (
        df.groupby("Subject_ID")["Label"].first().reset_index()
    )

    # 3) Convert string text labels into numeric values using LABEL_MAP
    subject_df["Label"] = subject_df["Label"].map(LABEL_MAP)

    # 4) Check for any unmapped or missing labels
    if subject_df["Label"].isna().any():
        print("\nERROR: Unmapped labels found!")
        print(subject_df[subject_df["Label"].isna()])
        return

    #extract arrays of subject IDs and their corresponding numeric labels
    subjects = subject_df["Subject_ID"].values
    labels   = subject_df["Label"].values.astype(int)

    print(f"\nTotal subjects: {len(subjects)}")
    print("Distribution:", Counter(labels))

    #split data into training/validation set (80%) and test set (20%) while maintaning class proportions (stratify) 
    s_trainval, s_test, y_trainval, y_test = train_test_split(
        subjects, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=SEED
    )

    #recalculate validation ratio relative to the remaining traival set
    effective_val_ratio = VAL_RATIO / (1 - TEST_RATIO)

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
        with open(OUT_DIR / f"{name}_subjects.txt", "w") as f:
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

            #get the numeric label for the current subject
            numeric_label = subject_df.loc[subject_df["Subject_ID"] == sid, "Label"].iloc[0]
            #reverse map numeric label back to its string folder name
            label_name = [k for k, v in LABEL_MAP.items() if v == numeric_label][0]
            #define relative path where the preprocessed .npy file is stored
            rel_path = f"{label_name}/{sid}.npy"   
            rows.append([sid, numeric_label, rel_path])

        #create DataFrame and save to CSV
        out_df = pd.DataFrame(rows, columns=["Subject_ID", "Label", "rel_path"])
        out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv → {len(out_df)} subjects")

    build_split_csv("train", s_train)
    build_split_csv("val",   s_val)
    build_split_csv("test",  s_test)

    # 6) Report and print class distributions for each split
    def report(name, y):
        c = Counter(y)
        print(
            f"{name:5s}: total {len(y)} | " +
            " | ".join(f"{lbl}:{cnt}" for lbl, cnt in sorted(c.items()))
        )

    print("\n=== SUBJECT-LEVEL DISTRIBUTIONS ===")
    report("Train", y_train)
    report("Val",   y_val)
    report("Test",  y_test)

    print("\nDONE — 3-class subject-level splits created successfully!\n")


main()
