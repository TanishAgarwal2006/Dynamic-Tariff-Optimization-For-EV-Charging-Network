# Adaptive Dynamic Pricing ML Framework for EV Charging Networks

## Overview & Workflow
This repository contains a closed-loop dynamic pricing pipeline designed to maximize Electric Vehicle (EV) charging station profitability while actively mitigating grid congestion. 

The framework operates through three distinct, continuously interacting modules:
* **Module 1: Demand Prediction Engine (CatBoost)** - Forecasts hourly charging volume, occupancy, and duration across urban charging zones.
* **Module 2: Mathematical Tariff Optimizer** - Dynamically adjusts service tariffs by balancing profit maximization against an exponential congestion penalty.
* **Module 3: Continuous Learning & Monitoring Module** - Utilizes an Exponential Moving Average (EMA) to extract the true Price Elasticity of Demand (PED) from day-over-day market variance, autonomously updating the system's assumptions to bridge the simulation-to-reality gap.

---

## Final Results (Month 6 Blind Test Evaluation)

Evaluation performed on the blind Month-6 (February 2023) test set.

### Module 1 — Demand Prediction Performance

| Target                        | Metric  |     Score |
| ----------------------------- | ------- | --------: |
| **Occupancy** | RMSE    |    2.0547 |
|                               | MAE     |    1.0461 |
|                               | R²      |  **0.9902** |
| **Volume** | RMSE    |  181.1081 |
|                               | MAE     |   43.9232 |
|                               | R²      |  **0.9547** |
| **Duration** | RMSE    |    1.7291 |
|                               | MAE     |    0.8983 |
|                               | R²      |  **0.9894** |
| **Congestion Classification** | ROC-AUC |  **0.9975** |

### Module 2 — Tariff Optimization Performance

| Metric                     | Improvement |
| -------------------------- | ----------: |
| Net Revenue Gain           | **+22.8829%** |
| Charger Utilization        |  **+0.0241%** |
| Off-Peak Charging Duration | **+10.3442%** |

### Module 3 — Monitoring & Learning Performance

| Metric                                   |           Improvement |
| ---------------------------------------- | --------------------: |
| Service Margin Efficiency (Yuan/kWh)     |          **+6.2652%** |
| Congestion Exposure (Waiting Time Proxy) | **+10.36% Reduction** |
| Customer Response Rate (Volume Shift)    |            **+1.99%** |
| Learned Average Market Elasticity        |               -1.6803 |

---

## Datasets
This project leverages two distinct datasets for exploratory analysis and model training:

* **ACN-Data:** Used exclusively for initial Exploratory Data Analysis (EDA) to understand session-level charging behavior. It was not used for model training as it lacks the critical dynamic pricing data required for this framework.
* **UrbanEV Dataset:** The primary dataset used for training the predictive models and simulating the dynamic pricing environment.

**Downloading UrbanEV Data:**
To run this project, you must manually download the core dataset files from the [IntelligentSystemsLab/UrbanEV GitHub repository](https://github.com/IntelligentSystemsLab/UrbanEV). 

Download the following 6 files and place them precisely inside the `data/raw/UrbanEV/` directory:
1.  `duration.csv`
2.  `e_price.csv`
3.  `inf.csv`
4.  `occupancy.csv`
5.  `s_price.csv`
6.  `volume.csv`

---

## Setup & Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TanishAgarwal2006/Adaptive-Dynamic-Pricing-ML-Framework-for-EV-Charging-Network
    cd Adaptive-Dynamic-Pricing-ML-Framework-for-EV-Charging-Network
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Exploratory Data Analysis (Optional):**
    * Run `preprocessing/eda_acn.ipynb` and `preprocessing/eda_urban.ipynb` to view the initial data distributions and spatial-temporal trends.
4.  **Data Preprocessing & Feature Engineering:**
    * Execute `preprocessing/preprocess_urbanEV.py` to clean the raw data.
    * Execute `preprocessing/featureEngg_urbanev.py` to generate the final temporal and spatial features for model ingestion.
5.  **Execute the Framework Pipeline:** Run the modules sequentially:
    * Train/Run Demand Predictor: `python src/agents/demand_agent.py`
    * Run Tariff Optimizer: `python src/agents/pricing_agent.py`
    * Run Feedback & Learning Loop: `python src/agents/learning_agent.py`

---

## Repository Structure

```text
C:.
│   requirements.txt
│   
├───configs
│       elasticity_memory.json
│       
├───data
│   ├───interim
│   │       urbanev_fused.csv
│   │       
│   ├───processed
│   │       acn_sessions_clean.csv
│   │       check.py
│   │       urbanev_features.csv
│   │       
│   └───raw
│       │   acndata_sessions.json
│       │   
│       └───UrbanEV
│               adfa.py
│               duration.csv
│               e_price.csv
│               inf.csv
│               occupancy.csv
│               s_price.csv
│               volume.csv
│               
├───models
│       demand_agent_congestion.cbm
│       demand_agent_duration.cbm
│       demand_agent_occupancy.cbm
│       demand_agent_volume.cbm
│       
├───preprocessing
│       demand.ipynb
│       eda_acn.ipynb
│       eda_urban.ipynb
│       featureEngg_urbanev.py
│       preprocess_acn.ipynb
│       preprocess_urbanEV.py
│       
├───results
│   │   elasticity_history.csv
│   │   learning_variance_history.csv
│   │   results.md
│   │   
│   └───figures
│           elasticity_learning_curve.png
│           queue_mitigation.png
│           revenue_comparison.png
│           
└───src
    │   generate_visuals.py
    │   
    └───agents
            demand_agent.py
            learning_agent.py
            pricing_agent.py
```

## Technical Implementation & Pipeline Workflow

The framework processes raw data into actionable, dynamic pricing tariffs through a carefully structured pipeline. Below is the technical breakdown of each file and module.

### 1. Exploratory Data Analysis & Preprocessing
* **EDA (`preprocessing/eda_urban.ipynb`):** Conducted macroscopic spatial-temporal analysis to identify city-wide charging trends. We analyzed peak occupancy hours, daily volume cycles, and evaluated historical correlations between service prices and charging demand.
* **Data Cleaning (`preprocessing/preprocess_urbanEV.py`):** The raw UrbanEV dataset consisted of disjointed matrices. This script successfully unpivoted (melted) the hour/day columns into a clean, flat time-series format and merged `volume`, `occupancy`, `duration`, `e_price`, and `s_price` into a cohesive master dataset.
* **Feature Engineering (`preprocessing/featureEngg_urbanev.py`):** Extracted granular temporal features including `hour`, `day_of_week`, `is_weekend`, and `month` to allow the predictive models to capture cyclical human commuting and commercial charging behavior.
* **Model Benchmarking (`preprocessing/demand.ipynb`):** Evaluated multiple gradient boosting frameworks (XGBoost, LightGBM, CatBoost) for the prediction task. **CatBoost** was selected as the final production model due to its superior handling of categorical geographical variables (`zone_id`) and highest $R^2$ accuracy.
* **Data Splitting Strategy:** To prevent data leakage and rigorously test the pipeline, the dataset was split chronologically. The first **5 months** were used strictly for model training, while the final **1 month** (Month 6: February 2023) was isolated as a blind test set.

---

### 2. The Core Modules

**Module 1: Demand Prediction Engine (`src/agents/demand_agent.py`)**
* **Function:** Acts as the foresight of the system.
* **Workflow:** It ingests the temporal features of the blind Month 6 test set and generates baseline forecasts.
* **Outputs:** For every hour and zone, it predicts the continuous targets: Volume (`pred_vol`), Occupancy (`pred_occ`), and Duration (`pred_dur`).

**Module 2: Tariff Optimization Engine (`src/agents/pricing_agent.py`)**
* **Function:** The mathematical optimizer responsible for dynamic pricing.
* **Workflow:** It takes the predicted demand and historical baseline prices. Initially, it utilizes a static assumption of the Price Elasticity of Demand (PED) based on the time of day. It simulates 50 candidate service prices, looking for the optimal balance. 
* **Reward Function:** It maximizes Station Profit while applying a steep, exponential penalty ($k=3$) to any price point that causes the simulated occupancy to approach or exceed physical charger capacity.
* **Outputs:** The mathematically optimized service price (`opt_s_price`), alongside the newly shifted volume and occupancy numbers.

**Module 3: Continuous Learning & Monitoring Engine (`src/agents/learning_agent.py`)**
* **Function:** The feedback loop that prevents the system from getting stuck in an optimistic simulation.
* **Workflow:** Instead of learning from its own simulated outputs (which causes a simulation paradox), it compares day-over-day true market data to calculate how real drivers reacted to price shifts. 
* **Updates:** It uses an Exponential Moving Average (EMA) with a set learning rate to smoothly update the `elasticity_memory.json` brain, enabling Module 2 to make smarter, more accurate pricing decisions the next day.

---

## Artifacts & Results Directory
The `results/` folder stores the definitive proof of the system's learning and financial impact:
* **`elasticity_history.csv` & `learning_variance_history.csv`:** These files track the daily parameter drift of the AI. By analyzing the decreasing variance over the 28-day simulation, we mathematically prove that the system is converging on the true market elasticity rather than oscillating wildly.
* **`figures/revenue_comparison.png`:** A visual breakdown showing the hourly financial uplift generated by the AI compared to historical baseline revenues.
* **`figures/queue_mitigation.png`:** Demonstrates how the system successfully flattened the demand curve, pushing traffic away from peak congestion hours to off-peak periods.

---

## Output Summary & Key Observations
1.  **Closing the Simulation Gap:** The initial hand-tuned elasticity assumptions proved overly optimistic. The integration of Module 3 successfully grounded the system in reality, slightly lowering expected revenue but creating a highly stable, mathematically defendable pipeline.
2.  **Queue Eradication:** By tracking "Unclipped Demand," we observed that the exponential penalty function in Module 2 successfully priced out grid-crushing demand, resulting in a near total reduction of physical waiting queues.
3.  **Margin Efficiency over Raw Volume:** The system successfully demonstrated that it wasn't just selling "more cheap power." By achieving a +6.2% Service Margin Efficiency, the AI proved it extracted premium value during high-demand windows.

---

## Limitations & Assumptions
* **Offline Learning Environment:** Because this pipeline evaluates historical 2023 data, Module 3 cannot run live A/B pricing tests. It relies on extracting elasticity signals from natural, historical day-over-day price variance.
* **Elasticity Isolation:** The framework operates under the assumption that day-over-day volume shifts are primarily driven by price (PED). It temporarily proxies out unmeasured external factors, such as extreme weather events or local traffic accidents.
* **Homogeneous Capacity Capping:** The queue mitigation logic assumes a hard limit on charger counts per zone without explicitly differentiating between heterogeneous charger speeds (e.g., Level 2 vs. DC Fast Chargers) in the capacity constraints.
