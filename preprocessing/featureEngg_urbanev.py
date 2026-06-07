import pandas as pd

def engineer_features(input_path, output_path):
    """
    Transforms the fused dataset into a supervised machine learning dataset.
    Creates explicit T+1 targets and uses Time T (and historical lags) as predictors.
    """
    print("Loading interim dataset...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort rigorously to ensure chronological shifting per zone
    df.sort_values(by=['zone_id', 'timestamp'], inplace=True)
    grouped = df.groupby('zone_id')
    
    print("Extracting temporal features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print("Creating Explicit Forecasting Targets (Time T+1)...")
    # These are your 'y' variables for model training
    df['target_occ'] = grouped['occupancy'].shift(-1)
    df['target_vol'] = grouped['volume'].shift(-1)
    df['target_occ_ratio'] = grouped['occ_ratio'].shift(-1)
    df['target_duration'] = grouped['duration'].shift(-1)
    
    print("Creating lag features for predictors (Time T-1, T-2, T-24, T-168)...")
    # Since the row itself represents Time T, shift(1) represents 1 hour ago
    for lag in [1, 2, 24, 168]:
        df[f'occ_lag_{lag}'] = grouped['occupancy'].shift(lag)
        df[f'occ_ratio_lag_{lag}'] = grouped['occ_ratio'].shift(lag)
        df[f'vol_lag_{lag}'] = grouped['volume'].shift(lag)
        df[f'duration_lag_{lag}'] = grouped['duration'].shift(lag)
            
    print("Creating rolling trend features (Ending at Time T)...")
    # 3-hour and 6-hour rolling averages of the current occupancy state
    df['occ_rolling_3h'] = grouped['occupancy'].transform(lambda x: x.rolling(window=3).mean())
    df['occ_rolling_6h'] = grouped['occupancy'].transform(lambda x: x.rolling(window=6).mean())
    
    print("Creating interaction features (Time T)...")
    # Baseline profit margin at the current hour
    df['price_spread'] = df['s_price'] - df['e_price']
    
    print("Cleaning up NaNs introduced by lag/target shifts...")
    # shift(168) creates 168 rows of NaNs at the beginning of each zone's history.
    # shift(-1) creates 1 row of NaNs at the very end of each zone's history.
    df.dropna(inplace=True)
    
    print("Saving final processed dataset...")
    df.to_csv(output_path, index=False)
    print(f"Dataset ready for Agent 1 training at: {output_path}")
    
    return df

if __name__ == "__main__":
    INTERIM_DATA_PATH = "data/interim/urbanev_fused.csv"
    PROCESSED_DATA_PATH = "data/processed/urbanev_features.csv"
    
    final_df = engineer_features(INTERIM_DATA_PATH, PROCESSED_DATA_PATH)
    
    print("\nSupervised Learning Snippet (Features at Time T -> Target at Time T+1):")
    cols_to_show = ['timestamp', 'zone_id', 'occupancy', 'occ_lag_1', 'target_occ']
    print(final_df[cols_to_show].head())