import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MonitoringLearningAgent:
    def __init__(self, config_dir="configs/", learning_rate=0.1, reset_memory=False):
        """
        Initializes the Learning Agent.
        """
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
        self.memory_path = os.path.join(self.config_dir, "elasticity_memory.json")
        self.results_dir = "results"
        self.figures_dir = os.path.join(self.results_dir, "figures")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.history_path = os.path.join(
            self.results_dir,
            "elasticity_history.csv"
        )
        self.learning_rate = learning_rate
        self.history_data = []

        # Explicit reset mechanism for reproducibility
        if reset_memory and os.path.exists(self.memory_path):
            os.remove(self.memory_path)
            print("Notice: Old elasticity memory deleted. Hard reset applied.")

        self.elasticity_memory = self._load_or_initialize_memory()

    def _load_or_initialize_memory(self):
        """Creates the baseline assumptions if no memory exists."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        
        memory = {}
        for hour in range(24):
            if 7 <= hour <= 10: e = -0.35
            elif 11 <= hour <= 16: e = -0.50
            elif 17 <= hour <= 20: e = -0.30
            else: e = -3.00 # The tuned late-night assumption
            memory[str(hour)] = e
            
        with open(self.memory_path, 'w') as f:
            json.dump(memory, f, indent=4)
        return memory

    def load_memory(self):
        with open(self.memory_path, 'r') as f:
            return json.load(f)

    def save_memory(self, memory):
        with open(self.memory_path, 'w') as f:
            json.dump(memory, f, indent=4)

    def log_history(self, day):
        memory = self.load_memory()
        record = {'day': day}
        record.update({f'hour_{h}': memory[str(h)] for h in range(24)})
        self.history_data.append(record)

    def export_history_csv(self):
        df = pd.DataFrame(self.history_data)
        df.to_csv(self.history_path, index=False)
        print(f"\nElasticity history saved to {self.history_path}")
        return df

    def learn_from_reality(self, today_df, yesterday_df):
        """
        Calculates TRUE historical elasticity at the ZONE LEVEL and updates the EMA.
        """
        memory = self.load_memory()
        total_zones_learned = 0
        
        for hour in range(24):
            t_data = today_df[today_df['hour'] == hour]
            y_data = yesterday_df[yesterday_df['hour'] == hour]
            
            if t_data.empty or y_data.empty:
                continue
                
            # MERGE ON ZONE_ID: Compare the exact same stations day-over-day
            merged = pd.merge(t_data, y_data, on='zone_id', suffixes=('_t', '_y'))
            if merged.empty:
                continue

            # Calculate total prices (s_price + e_price) for both days
            merged['price_t'] = merged['s_price_t'] + merged['e_price_t']
            merged['price_y'] = merged['s_price_y'] + merged['e_price_y']

            # Calculate % changes per zone
            merged['pct_change_price'] = (merged['price_t'] - merged['price_y']) / (merged['price_y'] + 1e-5)
            merged['pct_change_vol'] = (merged['volume_t'] - merged['volume_y']) / (merged['volume_y'] + 1e-5)

            # FILTER: Only keep zones where the price ACTUALLY changed by >= 2%
            valid_zones = merged[merged['pct_change_price'].abs() >= 0.02].copy()

            if valid_zones.empty:
                continue # No price variance across any zone for this hour
                
            total_zones_learned += len(valid_zones)

            # Calculate raw elasticity for the valid zones
            valid_zones['raw_e'] = valid_zones['pct_change_vol'] / valid_zones['pct_change_price']
            
            # Bound the physical reality (Elasticity should be between 0 and -5.0)
            valid_zones['raw_e'] = valid_zones['raw_e'].clip(lower=-5.0, upper=0.0)

            # AGGREGATE: Take the median of the VALID signals
            hourly_raw_e = valid_zones['raw_e'].median()

            # EMA Update Loop
            old_e = memory[str(hour)]
            new_e = (old_e * (1 - self.learning_rate)) + (hourly_raw_e * self.learning_rate)
            memory[str(hour)] = new_e
            
        self.save_memory(memory)
        
        # Feedback to the terminal to ensure learning is happening
        if total_zones_learned > 0:
            print(f"Agent 3: Extracted valid elasticity signals from {total_zones_learned} localized zone events.")

    def plot_learning_curve(self):
        df = pd.DataFrame(self.history_data)
        if df.empty: return
            
        plt.figure(figsize=(12, 6))
        hours_to_plot = [2, 9, 14, 18] 
        labels = ['2 AM (Night)', '9 AM (Morning Peak)', '2 PM (Midday)', '6 PM (Evening Peak)']
        colors = ['blue', 'orange', 'green', 'red']
        
        for h, label, col in zip(hours_to_plot, labels, colors):
            plt.plot(df['day'], df[f'hour_{h}'], label=label, linewidth=2, color=col)
            
        plt.title('Agent 3 Learning Curve: Adapting Price Elasticity (Month 6)')
        plt.xlabel('Day of Month')
        plt.ylabel('Price Elasticity of Demand (E)')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join( self.figures_dir, "elasticity_learning_curve.png")
        
        plt.savefig(plot_path)
        print(f"Learning curve graph saved to {plot_path}")
        
    def calculate_advanced_metrics(self, full_month_eval_df):
        """
        Carefully computes Phase 2 Monitoring Metrics:
        1. Pricing Efficiency Score
        2. Average Waiting Time Reduction (Queue Mitigation)
        3. Customer Response Rate (Elasticity Proxy)
        """
        df = full_month_eval_df.copy()
        
        # --- 1. SERVICE MARGIN EFFICIENCY (Yuan/kWh) ---
        baseline_margin_revenue = (df['baseline_s_price'] * df['pred_vol']).sum()
        baseline_total_vol = df['pred_vol'].sum()
        baseline_pes = baseline_margin_revenue / max(1, baseline_total_vol)

        if 'opt_s_price' in df.columns and 'opt_vol' in df.columns:
            opt_margin_revenue = (df['opt_s_price'] * df['opt_vol']).sum()
            opt_total_vol = df['opt_vol'].sum()
        else:
            opt_margin_revenue = baseline_margin_revenue
            opt_total_vol = baseline_total_vol
            
        opt_pes = opt_margin_revenue / max(1, opt_total_vol)
        pes_improvement_pct = ((opt_pes - baseline_pes) / baseline_pes) * 100

        # --- 2. WAITING TIME PROXY (Congestion Exposure) ---
        # Since UrbanEV contains no explicit queue or arrival timestamps,
        # waiting time is approximated using average charger occupancy ratio.

        df['baseline_congestion'] = (
            df['pred_occ'] / df['charge_count']
        ).clip(upper=1.0)

        if 'opt_occ' in df.columns:
            df['optimized_congestion'] = (
                df['opt_occ'] / df['charge_count']
            ).clip(upper=1.0)
        else:
            df['optimized_congestion'] = df['baseline_congestion']

        baseline_congestion = df['baseline_congestion'].mean()
        optimized_congestion = df['optimized_congestion'].mean()

        if baseline_congestion > 0:
            awt_reduction = (
                (baseline_congestion - optimized_congestion)
                / baseline_congestion
            ) * 100
        else:
            awt_reduction = 0.0
        # --- 3. CUSTOMER RESPONSE RATE (Elasticity Proxy) ---
        vol_shift_pct = ((opt_total_vol - baseline_total_vol) / max(1, baseline_total_vol)) * 100
        
        memory = self.load_memory()
        avg_learned_ped = np.mean(list(memory.values()))

        # --- PRINT METRIC REPORT ---
        print("\n================ AGENT 3 SCORECARD ENGINE ================")
        print(f"1. Service Margin Efficiency (Base): {baseline_pes:.4f} Yuan/kWh")
        print(f"   Service Margin Efficiency (Opt):  {opt_pes:.4f} Yuan/kWh")
        print(f"   Efficiency Gain:                  {pes_improvement_pct:+.4f}%")
        print("-" * 58)
        print(f"2. Baseline Congestion Exposure:     {baseline_congestion:.4f}")
        print(f"   Optimized Congestion Exposure:    {optimized_congestion:.4f}")
        print(f"   Waiting Time Proxy Improvement:   {awt_reduction:+.2f}%")
        print("-" * 58)
        print(f"3. Customer Response Rate (Vol Shift):{vol_shift_pct:+.2f}%")
        print(f"   Final Learned Avg Market Elasticity:{avg_learned_ped:+.4f}")
        print("==========================================================")
        
        return {
            "pes_improvement_pct": pes_improvement_pct,
            "awt_reduction_pct": awt_reduction,
            "vol_shift_pct": vol_shift_pct
        }

    def compute_learning_variance(self):
        """
        Tracks parameter drift using a 7-day smoothed window to ignore weekend/weekday noise.
        """
        if len(self.history_data) < 2:
            print("Insufficient history data to compute learning variance.")
            return []

        df_history = pd.DataFrame(self.history_data)
        variance_records = []

        # Calculate daily raw absolute drift
        for i in range(1, len(df_history)):
            day_current = df_history.iloc[i]
            day_previous = df_history.iloc[i - 1]
            
            hours_current = np.array([day_current[f'hour_{h}'] for h in range(24)])
            hours_previous = np.array([day_previous[f'hour_{h}'] for h in range(24)])
            
            daily_mad = np.mean(np.abs(hours_current - hours_previous))
            
            variance_records.append({
                "day": day_current['day'],
                "raw_variance": daily_mad
            })
            
        df_var = pd.DataFrame(variance_records)
        
        # Apply 7-day rolling average to filter out weekly seasonality
        df_var['smoothed_variance'] = df_var['raw_variance'].rolling(window=7, min_periods=1).mean()
        
        var_csv_path = os.path.join( self.results_dir, "learning_variance_history.csv")
        df_var.to_csv(var_csv_path, index=False)
        
        # Validation checks utilizing the smoothed trend
        if len(df_var) >= 7:
            # Compare Week 1 average vs Final Week average
            initial_variance = df_var['smoothed_variance'].iloc[0:7].mean()
            final_variance = df_var['smoothed_variance'].tail(7).mean()
            
            print(f"Stability Audit: Early Variance ({initial_variance:.4f}) vs Late Variance ({final_variance:.4f})")
            if final_variance < initial_variance:
                print(" -> Success: Elasticity parameter convergence confirmed (Noise Filtered).")
            else:
                print(" -> Notice: Parameters are still adjusting or market data remains highly volatile.")
                
        return variance_records


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.agents.demand_agent import DemandPredictionAgent
    from src.agents.pricing_agent import TariffPricingAgent
    
    print("--- Initiating Day-by-Day Month 6 Simulation ---")

    PROCESSED_DATA_PATH = "data/processed/urbanev_features.csv"
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    test_df = df[df['timestamp'] >= '2023-02-01'].copy()
    test_df['day'] = test_df['timestamp'].dt.day

    agent1 = DemandPredictionAgent(model_dir="models/")
    agent1.load_models()
    agent2 = TariffPricingAgent(config_dir="configs/")
    
    # Initialize Agent 3 WITH HARD RESET to ensure clean tracking
    agent3 = MonitoringLearningAgent(config_dir="configs/", learning_rate=0.1, reset_memory=True)

    total_eval_df = []
    days_in_feb = sorted(test_df['day'].unique())
    
    for current_day in days_in_feb:
        print(f"\n--- Simulating Feb {current_day:02d}, 2023 ---")
        
        today_data = test_df[test_df['day'] == current_day].copy()
        
        # Agent 1 Inference
        X_today = today_data.drop(columns=[c for c in agent1.features_to_drop if c in today_data.columns], errors='ignore')
        if 'zone_id' in X_today.columns: X_today['zone_id'] = X_today['zone_id'].astype('category')
            
        today_eval = pd.DataFrame({
            'hour': today_data['hour'],
            'e_price': today_data['e_price'],
            'baseline_s_price': today_data['s_price'],
            'pred_vol': agent1.models['volume'].predict(X_today),
            'pred_occ': agent1.models['occupancy'].predict(X_today),
            'pred_dur': agent1.models['duration'].predict(X_today),
            'charge_count': today_data['charge_count']
        })
        
        total_eval_df.append(today_eval)
        
        # Log before learning
        agent3.log_history(current_day)
        
        # Learn from Reality
        if current_day > 1:
            yesterday_data = test_df[test_df['day'] == current_day - 1]
            agent3.learn_from_reality(today_data, yesterday_data)

    print("\n================ FINAL MONTH 6 RESULTS ================")
    full_month_eval = pd.concat(total_eval_df)
    
    # 1. Unpack BOTH variables returned by Agent 2
    processed_month_df, agent2_metrics = agent2.evaluate_blind_test_set(full_month_eval) 
    
    # 2. Trigger Agent 3 advanced tracking calculations using the DataFrame
    agent3.calculate_advanced_metrics(processed_month_df)
    agent3.compute_learning_variance()
    
    # 3. Export logs and generate visualization curves
    agent3.export_history_csv()
    agent3.plot_learning_curve()