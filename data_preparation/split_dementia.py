# Subject-level stratified split for the dementia stages dataset.
# Input:
#   Z244_merged_slices_dementia.csv
#
# Output:
#   splits_dementia/train.csv
#   splits_dementia/val.csv
#   splits_dementia/test.csv
#
# Format:
#   subject_id, label, filepath
#   where label = dementia stage folder names (Very_mild, Mild, Moderate)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

#define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH   = SCRIPT_DIR / "Z244_merged_slices_dementia.csv"

OUT_DIR = SCRIPT_DIR.parent / "splits_dementia"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#split configuration parameters
TEST_RATIO = 0.20
VAL_RATIO  = 0.20
SEED = 42

def main():

    # 1) Load the merged slices CSV file
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    assert "Subject_ID" in df.columns
    assert "Label" in df.columns

    # 2) Create a subject-level dataframe 
    subject_df = df.groupby("Subject_ID")["Label"].first().reset_index()

    #extract arrays of subject IDs and their corresponding string labels
    subjects = subject_df["Subject_ID"].values
    labels   = subject_df["Label"].values

    #handle moderate due to very few samples
    moderate_mask = labels == "Moderate"
    mod_subjects  = subjects[moderate_mask]

    #ensure there are exactly 2 Moderate subjects as expected
    if len(mod_subjects) != 2:
        raise ValueError(f"Expected 2 Moderate subjects, found {len(mod_subjects)}")

    #manually assign one Moderate subject to train and one to test
    mod_train = mod_subjects[0]
    mod_test  = mod_subjects[1]

    #remove Moderate subjects from the rest of the dataset before stratified splitting
    mask_rest = (subjects != mod_train) & (subjects != mod_test)
    rest_subjects = subjects[mask_rest]
    rest_labels   = labels[mask_rest]

    
    #split the remaining subjects into training/validation set and test set while maintaining class proportions
    s_trainval, s_test, y_trainval, y_test = train_test_split(
        rest_subjects, rest_labels,
        test_size=TEST_RATIO,
        stratify=rest_labels,
        random_state=SEED
    )

    #manually add the Moderate subject into the test set
    s_test = list(s_test) + [mod_test]
    y_test = list(y_test) + ["Moderate"]

    #recalculate validation ratio relative to the remaining trainval set
    eff_val = VAL_RATIO / (1.0 - TEST_RATIO)
    
    #further split trainval into actual training set and validation set
    s_train, s_val, y_train, y_val = train_test_split(
        s_trainval, y_trainval,
        test_size=eff_val,
        stratify=y_trainval,
        random_state=SEED
    )

    #manually add the other Moderate subject into the training set
    s_train = list(s_train) + [mod_train]
    y_train = list(y_train) + ["Moderate"]

    def save_txt(filename, lst):
        with open(OUT_DIR / f"{filename}_subjects.txt", "w") as f:
            for sid in lst:
                f.write(f"{sid}\n")

    save_txt("train", s_train)
    save_txt("val", s_val)
    save_txt("test", s_test)

  
    def build_csv(name, sids):
        rows = []
        for sid in sids:
            #get the string label for the current subject
            lbl = subject_df.loc[subject_df["Subject_ID"] == sid, "Label"].iloc[0]
            #define relative path where the preprocessed .npy file is stored 
            rel_path = f"{lbl}/{sid}.npy"   
            rows.append([sid, lbl, rel_path])

        #create DataFrame and save to CSV
        out_df = pd.DataFrame(rows, columns=["subject_id","label","filepath"])
        out_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv → {len(out_df)} subjects")

    build_csv("train", s_train)
    build_csv("val", s_val)
    build_csv("test", s_test)


    def report(name, labels):
        c = Counter(labels)
        print(f"{name:6}: " + ", ".join(f"{k}:{v}" for k,v in c.items()))

    print("\n=== FINAL SUBJECT DISTRIBUTION ===")
    report("Train", y_train)
    report("Val", y_val)
    report("Test", y_test)

    print("\nSaved splits to:", OUT_DIR)


main()
