import pandas as pd
from functools import reduce
import os

def load_and_melt(filepath, value_name):
    """Loads a wide-format CSV and melts it into a long-format DataFrame."""
    print(f"Melting {value_name}...")
    df = pd.read_csv(filepath)
    time_col = df.columns[0]
    
    df_melted = df.melt(id_vars=[time_col], var_name="zone_id", value_name=value_name)
    df_melted.rename(columns={time_col: "timestamp"}, inplace=True)
    df_melted["timestamp"] = pd.to_datetime(df_melted["timestamp"], format="mixed", dayfirst=True)
    df_melted["zone_id"] = df_melted["zone_id"].astype(int)
    
    return df_melted

def fuse_and_fix_data(data_dir):
    """Fuses time-series data and appends occ_ratio."""
    print("Processing zone metadata...")
    inf_df = pd.read_csv(os.path.join(data_dir, "inf.csv"))
    zone_meta = inf_df.groupby("TAZID")["charge_count"].sum().reset_index()
    zone_meta.rename(columns={"TAZID": "zone_id"}, inplace=True)
    
    files = {
        "occupancy": "occupancy.csv",
        "volume": "volume.csv",
        "duration": "duration.csv",
        "s_price": "s_price.csv",
        "e_price": "e_price.csv"
    }
    
    dataframes = []
    for value_name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        dataframes.append(load_and_melt(filepath, value_name))
        
    print("Merging time-series dataframes...")
    # Changed to 'outer' to prevent data loss from missing single-sensor readings
    master_ts_df = reduce(
        lambda left, right: pd.merge(left, right, on=["timestamp", "zone_id"], how="outer"), 
        dataframes
    )
    
    print("Calculating occ_ratio...")
    
    master_ts_df = pd.merge(master_ts_df, zone_meta, on="zone_id", how="left")
    master_ts_df["occ_ratio"] = master_ts_df["occupancy"] / master_ts_df["charge_count"]
    
    final_df = master_ts_df[[ "timestamp", "zone_id", "charge_count", "occupancy",  "occ_ratio", "volume", "duration", "s_price", "e_price"]].copy()
    final_df.sort_values(by=["zone_id", "timestamp"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    
    return final_df

if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw/UrbanEV"
    INTERIM_DATA_PATH = "data/interim/urbanev_fused.csv"
    
    final_df = fuse_and_fix_data(RAW_DATA_DIR)
    
    print("\nClean Master Dataframe Info:")
    print(final_df.info())
    final_df.to_csv(INTERIM_DATA_PATH, index=False)