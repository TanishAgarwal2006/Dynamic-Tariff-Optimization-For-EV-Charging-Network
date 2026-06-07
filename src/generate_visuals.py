import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Python can find your custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.demand_agent import DemandPredictionAgent
from src.agents.pricing_agent import TariffPricingAgent

def setup_directories():
    """Creates the output directory for the final report figures."""
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_month_6_data():
    """Runs the Month 6 batch inference to get the raw plotting data."""
    print("Loading data and generating batch predictions...")
    
    df = pd.read_csv("data/processed/urbanev_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    test_df = df[df['timestamp'] >= '2023-02-01'].copy()

    agent1 = DemandPredictionAgent(model_dir="models/")
    agent1.load_models()
    agent2 = TariffPricingAgent(config_dir="configs/")

    # Format for Agent 1
    X_test = test_df.drop(columns=[c for c in agent1.features_to_drop if c in test_df.columns], errors='ignore')
    if 'zone_id' in X_test.columns:
        X_test['zone_id'] = X_test['zone_id'].astype('category')

    # Agent 1 Inference
    eval_df = pd.DataFrame({
        'hour': test_df['hour'],
        'e_price': test_df['e_price'],
        'baseline_s_price': test_df['s_price'],
        'pred_vol': agent1.models['volume'].predict(X_test),
        'pred_occ': agent1.models['occupancy'].predict(X_test),
        'pred_dur': agent1.models['duration'].predict(X_test),
        'charge_count': test_df['charge_count']
    })

    # Agent 2 Optimization (We suppress the print statements for clean plotting)
    import sys, os as _os
    old_stdout = sys.stdout
    sys.stdout = open(_os.devnull, 'w')
    processed_df, _ = agent2.evaluate_blind_test_set(eval_df)
    sys.stdout = old_stdout

    print("Data generation complete. Rendering charts...")
    return processed_df

def plot_revenue_curve(df, output_dir):
    """Generates the 24-hour Revenue Comparison Chart."""
    # Aggregate data by hour
    hourly_data = df.groupby('hour')[['baseline_profit', 'optimized_profit']].mean().reset_index()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Plot lines
    ax.plot(hourly_data['hour'], hourly_data['baseline_profit'], 
            label='Baseline Historical Revenue', color='#7f8c8d', linewidth=2.5, linestyle='--')
    ax.plot(hourly_data['hour'], hourly_data['optimized_profit'], 
            label='AI Optimized Revenue', color='#27ae60', linewidth=3.5)

    # Fill the area between to highlight the profit gain
    ax.fill_between(hourly_data['hour'], hourly_data['baseline_profit'], hourly_data['optimized_profit'], 
                    where=(hourly_data['optimized_profit'] > hourly_data['baseline_profit']), 
                    interpolate=True, color='#2ecc71', alpha=0.2, label='Net Profit Gain')

    # Formatting
    ax.set_title('Average Daily Revenue Cycle (Month 6 Evaluation)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day (0-23)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Profit (Yuan)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'revenue_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"-> Saved: {save_path}")

def plot_queue_mitigation(df, output_dir):
    """Generates the 24-hour Queue Mitigation Chart showing peak flattening."""
    hourly_data = df.groupby('hour')[['pred_occ', 'opt_occ', 'charge_count']].mean().reset_index()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Plot capacity threshold
    ax.plot(hourly_data['hour'], hourly_data['charge_count'], 
            label='Physical Charger Capacity', color='#c0392b', linewidth=2, linestyle=':')

    # Plot Occupancy Lines
    ax.plot(hourly_data['hour'], hourly_data['pred_occ'], 
            label='Baseline Occupancy (Unmanaged Demand)', color='#e67e22', linewidth=2.5, linestyle='--')
    ax.plot(hourly_data['hour'], hourly_data['opt_occ'], 
            label='AI Managed Occupancy', color='#2980b9', linewidth=3.5)

    # Highlight danger zones visually
    ax.fill_between(hourly_data['hour'], hourly_data['pred_occ'], hourly_data['charge_count'], 
                    where=(hourly_data['pred_occ'] > hourly_data['charge_count']), 
                    interpolate=True, color='#e74c3c', alpha=0.3, label='Baseline Queue/Congestion')

    # Formatting
    ax.set_title('Grid Balancing & Queue Mitigation (Month 6 Evaluation)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day (0-23)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Vehicle Occupancy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'queue_mitigation.png')
    plt.savefig(save_path)
    plt.close()
    print(f"-> Saved: {save_path}")

if __name__ == "__main__":
    out_dir = setup_directories()
    results_df = generate_month_6_data()
    plot_revenue_curve(results_df, out_dir)
    plot_queue_mitigation(results_df, out_dir)
    print("\nAll presentation visuals generated successfully.")