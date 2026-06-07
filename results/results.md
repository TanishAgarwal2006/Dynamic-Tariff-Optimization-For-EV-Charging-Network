# Experiment Summary

Evaluation performed on the blind Month-6 (February 2023) test set.

Artifacts generated:
- Agent 1 predictive metrics
- Agent 2 optimization metrics
- Agent 3 monitoring metrics
- Elasticity learning history
- Learning variance history
- Learning curve visualization

---


### Agent 1 — Demand Prediction Performance

| Target                        | Metric  |    Score |
| ----------------------------- | ------- | -------: |
| **Occupancy**                 | RMSE    |   2.0547 |
|                               | MAE     |   1.0461 |
|                               | R²      |   **0.9902** |
| **Volume**                    | RMSE    | 181.1081 |
|                               | MAE     |  43.9232 |
|                               | R²      |   **0.9547** |
| **Duration**                  | RMSE    |   1.7291 |
|                               | MAE     |   0.8983 |
|                               | R²      |  **0.9894** |
| **Congestion Classification** | ROC-AUC |   **0.9975** |

---

### Agent 2 — Tariff Optimization Performance

| Metric                     | Improvement |
| -------------------------- | ------------: |
| Net Revenue Gain           | **+22.8829%** |
| Charger Utilization        |  **+0.0241%** |
| Off-Peak Charging Duration | **+10.3442%** |

---

### Agent 3 — Monitoring & Learning Performance

| Metric                                   |           Improvement |
| ---------------------------------------- | --------------------: |
| Service Margin Efficiency (Yuan/kWh)     |          **+6.2652%** |
| Congestion Exposure (Waiting Time Proxy) | **+10.36% Reduction** |
| Customer Response Rate (Volume Shift)    | **+1.99%** |
| Learned Average Market Elasticity        | -1.6803 |
