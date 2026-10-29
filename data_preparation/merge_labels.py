import os
import pandas as pd

INPUT_CSV  = "Z244_centered_slices.csv"     

#3-class classification scenario mapping (merging Mild and Moderate dementia categories)
CLASS_MAPPING_3 = {
    "Non Demented":       "Non_Demented",
    "Very mild Dementia": "Very_mild_Dementia",
    "Mild Dementia":      "Mild_Moderate_Dementia",
    "Moderate Dementia":  "Mild_Moderate_Dementia",
}

#binary classification scenario mapping (Non Demented vs. Demented categories)
CLASS_MAPPING_2 = {
    "Non Demented":       "Non_Demented",
    "Very mild Dementia": "Demented",
    "Mild Dementia":      "Demented",
    "Moderate Dementia":  "Demented",
}

#dementia-only scenario mapping (focusing exclusively on disease severity stages)
CLASS_MAPPING_DEMENTED_3 = {
    "Very mild Dementia": "Very_mild",
    "Mild Dementia":      "Mild",
    "Moderate Dementia":  "Moderate",
}

#choose strategy 
MODE = "dementia"   

#default empty set for labels to be excluded
EXCLUDE_OLD_LABELS = set()

if MODE == "3class":
    ACTIVE_MAPPING = CLASS_MAPPING_3
    OUTPUT_CSV     = "Z244_merged_slices_3class.csv"
elif MODE == "binary":
    ACTIVE_MAPPING = CLASS_MAPPING_2
    OUTPUT_CSV     = "Z244_merged_slices_binary.csv"
elif MODE == "dementia":
    ACTIVE_MAPPING = CLASS_MAPPING_DEMENTED_3
    #exclude healthy subjects when focusing solely on dementia severity stages
    EXCLUDE_OLD_LABELS = {"Non Demented"}
    OUTPUT_CSV         = "Z244_merged_slices_dementia.csv"
else:
    raise ValueError(f"Unknown MODE='{MODE}'")


def merge_labels(input_path: str, output_path: str, mapping: dict, exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()

    #read the centered slices metadata CSV 
    try:
        df = pd.read_csv(input_path, sep=r"\s{2,}", engine="python")
    except Exception:
        df = pd.read_csv(input_path)

    #find label column    
    label_col = None
    for col in df.columns:
        if "Label" in col or "class" in col.lower():
            label_col = col
            break

    if label_col is None:
        print("Label not found")
        return

    #drop rows where Label is NaN 
    before_nan = len(df)
    df = df.dropna(subset=[label_col])
    after_nan = len(df)
    if after_nan < before_nan:
        print(f"Dropped rows with NaN in Label")
    
    #filter out specific old labels if specified in the configuration
    if exclude_labels:
        before_excl = len(df)
        df = df[~df[label_col].isin(exclude_labels)].copy()
        after_excl = len(df)
        print(f"Dropped rows due to EXCLUDE_OLD_LABELS={exclude_labels}")

    #mapping, re-label the classes
    df[label_col] = df[label_col].map(mapping)

    #check for labels that didn't map
    if df[label_col].isna().any():
        missing = df[df[label_col].isna()]
        print("\nSome labels are missing:")
        print(missing[label_col].drop_duplicates())
        return

    #keep basic columns in order    
    base_cols = ["Subject_ID", label_col, "Z_index", "Filepath"]
    existing = [c for c in base_cols if c in df.columns]
    others   = [c for c in df.columns if c not in existing]
    df = df[existing + others]

    #save new csv
    df.to_csv(output_path, index=False)
    print(f"Saved at: {output_path}")

    #distribution per new class    
    if "Subject_ID" in df.columns:
        print("\nDistribution subject per new class:")
        print(df.groupby(label_col)["Subject_ID"].nunique())


def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    in_path     = os.path.join(script_dir, INPUT_CSV)
    out_path    = os.path.join(script_dir, OUTPUT_CSV)

    print(f"\nMerging labels with MODE='{MODE}':")
    merge_labels(
        input_path=in_path,
        output_path=out_path,
        mapping=ACTIVE_MAPPING,
        exclude_labels=EXCLUDE_OLD_LABELS
    )


main()
