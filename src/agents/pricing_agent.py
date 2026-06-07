import os
import json
import sys
import numpy as np
import pandas as pd

class TariffPricingAgent:
    
    def __init__(self, config_dir="configs/"):
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
        self.memory_path = os.path.join(self.config_dir, "elasticity_memory.json")
        
        self.elasticity_memory = self._load_or_initialize_memory()
        
        # Hyperparameters for the Objective Function
        self.penalty_weight = 500        # Beta: Penalty for grid congestion
        self.penalty_power = 3           # k: Exponential curve for penalty
        self.market_share_weight = 0.15  # Gamma: LOWERED to prevent profit cannibalization
    # ... [Keep _load_or_initialize_memory and _get_elasticity exactly as they are] ...


    def _load_or_initialize_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        
        memory = {}
        for hour in range(24):
            if 7 <= hour <= 10:
                e = -0.35  # Morning Peak
            elif 11 <= hour <= 16:
                e = -0.50  # Midday
            elif 17 <= hour <= 20:
                e = -0.30  # Evening Peak
            else:
                e = -3.0  # Night (21-6): Highly Elastic (Commercial/Fleet behavior)
            
            memory[str(hour)] = e
            
        with open(self.memory_path, 'w') as f:
            json.dump(memory, f, indent=4)
        return memory

    def _get_elasticity(self, hour):
        return self.elasticity_memory[str(hour)]


    def optimize_tariff(self, hour, e_price, baseline_s_price, pred_vol, pred_occ, charge_count):
        current_E = self._get_elasticity(hour)
        total_baseline_price = baseline_s_price + e_price
        
        if pred_vol <= 0:
            return baseline_s_price, 1.0, 0, min(pred_occ, charge_count)
            
        pred_utilization = pred_occ / max(1, charge_count)
        is_off_peak = (hour >= 21) or (hour <= 6)
        is_peak = (17 <= hour <= 20)
        
        # --- TUNED BUSINESS RULES ---
        if pred_utilization < 0.10:
            # Rule 1: Truly Empty Grid (Bottom 10%). Deep discounts allowed.
            min_search = max(0.01, baseline_s_price * 0.40)
            max_search = max(0.02, baseline_s_price * 0.90) 
        elif is_off_peak:
            # Rule 2: Nighttime (but >10% util). Cap at baseline.
            min_search = max(0.01, baseline_s_price * 0.60)
            max_search = baseline_s_price  
        elif is_peak:
            # Rule 3: RUSH HOUR. No discounts allowed. Surge up to 200%.
            min_search = baseline_s_price
            max_search = min(1.45, baseline_s_price * 2.00)
        else:
            # Rule 4: Normal Daytime. Moderate adjustments.
            min_search = max(0.01, baseline_s_price * 0.80)
            max_search = min(1.45, baseline_s_price * 1.50)
            
        candidate_prices = np.linspace(min_search, max_search, num=50)
        
        best_s_price = baseline_s_price
        best_score = -np.inf
        best_sim_vol = pred_vol
        best_sim_occ = pred_occ
        best_multiplier = 1.0
        
        for candidate_s_price in candidate_prices:
            total_candidate_price = candidate_s_price + e_price
            price_pct_change = (total_candidate_price - total_baseline_price) / (total_baseline_price + 1e-5)
            
            demand_multiplier = max(0.0, 1.0 + (current_E * price_pct_change))
            
            sim_vol = pred_vol * demand_multiplier
            sim_occ = pred_occ * demand_multiplier
            
            # --- THE REWARD FUNCTION ---
            # 1. Profit
            sim_profit = sim_vol * candidate_s_price
            
            # 2. Market Share Bonus (Only active when grid is truly hurting for volume)
            if is_off_peak or pred_utilization < 0.15:
                market_share_bonus = sim_vol * self.market_share_weight
            else:
                market_share_bonus = 0
                
            # 3. Congestion Penalty
            occ_ratio = sim_occ / max(1, charge_count) 
            penalty = self.penalty_weight * (occ_ratio ** self.penalty_power)
            
            # Final AI Objective
            score = sim_profit + market_share_bonus - penalty
            
            if score > best_score:
                best_score = score
                best_s_price = candidate_s_price
                best_sim_vol = sim_vol
                best_sim_occ = sim_occ
                best_multiplier = demand_multiplier
                
        final_physical_occ = min(best_sim_occ, charge_count)
                
        return best_s_price, best_multiplier, best_sim_vol, final_physical_occ
        
    def evaluate_blind_test_set(self, test_df):
            """
            Part 2: Executes full batch evaluation over the evaluation set and computes core system metrics.
            Now explicitly returns the processed DataFrame for Agent 3 to analyze.
            """
            print("\n--- Running Agent 2 Optimization over Test Set ---")
            
            optimized_records = []
            
            for idx, row in test_df.iterrows():
                hour = int(row['hour'])
                e_price = row['e_price']
                base_s_price = row['baseline_s_price']
                pred_vol = row['pred_vol']
                pred_occ = row['pred_occ']
                pred_dur = row['pred_dur']
                charge_count = max(1, int(row['charge_count']))
                
                # Run Optimization Engine
                opt_s_price, demand_multiplier, sim_vol, sim_occ = self.optimize_tariff(
                    hour=hour,
                    e_price=e_price,
                    baseline_s_price=base_s_price,
                    pred_vol=pred_vol,
                    pred_occ=pred_occ,
                    charge_count=charge_count
                )
                
                sim_dur = min(pred_dur * demand_multiplier, charge_count * 1.0)
                baseline_profit = pred_vol * base_s_price
                optimized_profit = sim_vol * opt_s_price
                
                # NEW: Log ALL features so Agent 3 has the raw data to calculate Efficiency & Queues
                optimized_records.append({
                    'hour': hour,
                    'charge_count': charge_count,
                    'e_price': e_price,
                    'baseline_s_price': base_s_price,
                    'opt_s_price': opt_s_price,
                    'pred_vol': pred_vol,
                    'opt_vol': sim_vol,
                    'pred_occ': pred_occ,
                    'opt_occ': sim_occ,
                    'baseline_profit': baseline_profit,
                    'optimized_profit': optimized_profit,
                    'baseline_dur': pred_dur,
                    'optimized_dur': sim_dur
                })
                
            res_df = pd.DataFrame(optimized_records)
            
            # --- Metrics Calculation ---
            total_base_profit = res_df['baseline_profit'].sum()
            total_opt_profit = res_df['optimized_profit'].sum()
            revenue_gain_pct = ((total_opt_profit - total_base_profit) / (total_base_profit + 1e-5)) * 100
            
            total_capacity_hours = (res_df['charge_count'] * 1.0).sum()
            baseline_utilization = (res_df['baseline_dur'].sum() / total_capacity_hours) * 100
            optimized_utilization = (res_df['optimized_dur'].sum() / total_capacity_hours) * 100
            
            off_peak_mask = (res_df['hour'] >= 21) | (res_df['hour'] <= 6)
            off_peak_df = res_df[off_peak_mask]
            
            base_off_peak_dur = off_peak_df['baseline_dur'].sum()
            opt_off_peak_dur = off_peak_df['optimized_dur'].sum()
            off_peak_uplift_pct = ((opt_off_peak_dur - base_off_peak_dur) / (base_off_peak_dur + 1e-5)) * 100
            
            print("\n================ AGENT 2 EVALUATION METRICS ================")
            print(f"1. Net Revenue Gain:          {revenue_gain_pct:+.4f}%")
            print(f"2. Baseline Charger Util:     {baseline_utilization:.4f}%")
            print(f"   Optimized Charger Util:    {optimized_utilization:.4f}%")
            print(f"3. Off-Peak Uplift Duration:  {off_peak_uplift_pct:+.4f}%")
            print("============================================================\n")
            
            metrics_dict = {
                'revenue_gain_pct': revenue_gain_pct,
                'baseline_utilization': baseline_utilization,
                'optimized_utilization': optimized_utilization,
                'off_peak_uplift_pct': off_peak_uplift_pct
            }
            
            # NEW: Return BOTH the DataFrame and the Dictionary
            return res_df, metrics_dict


if __name__ == "__main__":
    
    # Ensure Python can find the src module to import Agent 1
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.agents.demand_agent import DemandPredictionAgent
    
    print("--- Initiating End-to-End Test Set Evaluation ---")

    # 1. Load the REAL processed dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    PROCESSED_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR,"..","..","data","processed","urbanev_features.csv"))
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Could not find {PROCESSED_DATA_PATH}. Please run preprocessing first.")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Isolate the blind test set (Month 6: February 2023)
    test_split_date = '2023-02-01'
    test_df = df[df['timestamp'] >= test_split_date].copy()
    print(f"Isolated Month 6 Test Set: {len(test_df)} records.")

    # 3. Load Agent 1 to generate REAL forecasts
    print("Loading Agent 1 (Demand Predictor) to generate forecasts...")
    agent1 = DemandPredictionAgent(model_dir="models/")
    agent1.load_models()

    # Format the inference matrix for Agent 1 (drop targets)
    X_test = test_df.drop(columns=[c for c in agent1.features_to_drop if c in test_df.columns], errors='ignore')
    if 'zone_id' in X_test.columns:
        X_test['zone_id'] = X_test['zone_id'].astype('category')

    # 4. Construct the Evaluation DataFrame for Agent 2
    print("Formatting prediction pipeline for Agent 2...")
    mean_s_price = test_df['s_price'].mean()
    eval_df = pd.DataFrame({
        'hour': test_df['hour'],
        'e_price': test_df['e_price'],
        'baseline_s_price': mean_s_price, # The historical baseline we are trying to beat
        'pred_vol': agent1.models['volume'].predict(X_test),
        'pred_occ': agent1.models['occupancy'].predict(X_test),
        'pred_dur': agent1.models['duration'].predict(X_test),
        'charge_count': test_df['charge_count']
    })

    # 5. Run Agent 2 on the REAL predicted data
    agent2 = TariffPricingAgent(config_dir="configs/")
    metrics = agent2.evaluate_blind_test_set(eval_df)