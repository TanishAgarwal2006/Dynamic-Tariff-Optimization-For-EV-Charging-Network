import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import os

class DemandPredictionAgent:
    def __init__(self, model_dir="models/", congestion_threshold=0.85):
        """
        Initializes the Multi-Target Demand Prediction Agent.
        """
        self.model_dir = model_dir
        self.congestion_threshold = congestion_threshold
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Base parameters for all CatBoost models
        base_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'cat_features': ['zone_id'],
            'random_seed': 42,
            'thread_count': -1,
            'verbose': False # Silenced to keep multi-training terminal output clean
        }
        
        # Initialize the 3 Regressors and 1 Classifier
        self.models = {
            'occupancy': CatBoostRegressor(**base_params),
            'volume': CatBoostRegressor(**base_params),
            'duration': CatBoostRegressor(**base_params),
            'congestion': CatBoostClassifier(**base_params)
        }
        
        self.features_to_drop = [
            'timestamp', 'target_occ', 'target_vol', 
            'target_duration', 'target_occ_ratio', 'target_congestion'
        ]

    def _split_data(self, df):
        val_split_date = '2023-01-01'
        test_split_date = '2023-02-01'

        train_df = df[df['timestamp'] < val_split_date].copy()
        val_df = df[(df['timestamp'] >= val_split_date) & (df['timestamp'] < test_split_date)].copy()
        test_df = df[df['timestamp'] >= test_split_date].copy()

        return train_df, val_df, test_df

    def train(self, data_path):
        print("Loading and preparing dataset...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['zone_id'] = df['zone_id'].astype('category')
        
        # Dynamically create the binary congestion target
        df['target_congestion'] = (df['target_occ_ratio'] >= self.congestion_threshold).astype(int)

        train_df, val_df, test_df = self._split_data(df)

        # Prepare base X matrices
        X_train = train_df.drop(columns=[c for c in self.features_to_drop if c in train_df.columns])
        X_val = val_df.drop(columns=[c for c in self.features_to_drop if c in val_df.columns])
        self.X_test = test_df.drop(columns=[c for c in self.features_to_drop if c in test_df.columns])

        # Define targets mapping
        targets = {
            'occupancy': 'target_occ',
            'volume': 'target_vol',
            'duration': 'target_duration',
            'congestion': 'target_congestion'
        }
        
        # Store test targets for evaluation
        self.y_test_dict = {name: test_df[col] for name, col in targets.items()}

        # Train each model independently
        for name, model in self.models.items():
            print(f"Training {name} model...")
            target_col = targets[name]
            y_train = train_df[target_col]
            y_val = val_df[target_col]
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50
            )
            
        self.save_models()
        print("All models trained successfully.")

    def evaluate(self):
        print("\n--- Blind Test Set Evaluation (Month 6) ---")
        metrics = {}
        
        for name, model in self.models.items():
            y_true = self.y_test_dict[name]
            y_pred = model.predict(self.X_test)
            
            if name == 'congestion':
                # For classification, check ROC-AUC
                # predict_proba returns [prob_0, prob_1], we want prob_1
                y_prob = model.predict_proba(self.X_test)[:, 1]
                auc = roc_auc_score(y_true, y_prob)
                print(f"[{name.upper()}] ROC-AUC: {auc:.4f}")
                metrics[name] = {"roc_auc": auc}
            else:
                # For regression, check standard metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"[{name.upper()}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
                metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}
                
        return metrics

    def predict(self, current_state_df, zone_charge_count):
        """
        Takes current hour features and static capacity to return all 4 metrics.
        """
        if 'zone_id' in current_state_df.columns:
            current_state_df['zone_id'] = current_state_df['zone_id'].astype('category')
            
        X_infer = current_state_df.drop(columns=[c for c in self.features_to_drop if c in current_state_df.columns], errors='ignore')
        
        predictions = {
            'predicted_occupancy': self.models['occupancy'].predict(X_infer),
            'predicted_volume': self.models['volume'].predict(X_infer),
            'predicted_duration_hrs': self.models['duration'].predict(X_infer)
        }
        
        # Calculate Utilization %
        predictions['predicted_utilization_pct'] = (predictions['predicted_duration_hrs'] / zone_charge_count) * 100
        
        # Extract the probability of hitting class '1' (Congested)
        predictions['congestion_probability'] = self.models['congestion'].predict_proba(X_infer)[:, 1]
        
        return predictions

    def save_models(self):
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f"demand_agent_{name}.cbm")
            model.save_model(path)

    def load_models(self):
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f"demand_agent_{name}.cbm")
            model.load_model(path)
            
if __name__ == "__main__":
    # Define the path to your fully engineered dataset
    PROCESSED_DATA_PATH = "data/processed/urbanev_features.csv"
    
    print("Initializing Demand Prediction Agent...")
    agent = DemandPredictionAgent(congestion_threshold=0.85)
    
    # Train and save the models
    agent.train(PROCESSED_DATA_PATH)
    
    # Evaluate on the blind test set
    metrics = agent.evaluate()
    print("\nFinal Model Evaluation Complete.")