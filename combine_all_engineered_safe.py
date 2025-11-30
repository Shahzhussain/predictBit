# combine_all_engineered_safe.py
import os
import glob
import pandas as pd

BASE = ".."
ENG_DIR = os.path.join(BASE, "data_engineered")
OUT_PATH = os.path.join(ENG_DIR, "final_master_dataset.csv")

def combine_all_engineered(output_path=OUT_PATH):
    pattern = os.path.join(ENG_DIR, "engineered_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No engineered files found in", ENG_DIR)
        return
    dfs = []
    for f in files:
        print("Reading", f)
        df = pd.read_csv(f)
        # ensure date parsed
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)
    master = pd.concat(dfs, ignore_index=True, sort=False)
    master = master.sort_values("date", ascending=True).reset_index(drop=True)
    master.to_csv(output_path, index=False)
    print("Saved final master dataset:", output_path)
    print("Rows:", master.shape[0])

if __name__ == "__main__":
    combine_all_engineered()
