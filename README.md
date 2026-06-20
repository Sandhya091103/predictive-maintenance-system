# Predictive Maintenance System

An end-to-end machine learning system that predicts industrial equipment failures using IoT sensor data — enabling proactive maintenance scheduling and reducing unplanned downtime.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.9% |
| Precision | 82.9% |
| Recall | 85.3% |
| F1 Score | 0.84 |
| Best Model | XGBoost |
| Optimized Threshold | 0.35 |

- **10,000** sensor readings analyzed
- **339** failure events detected (3.39% failure rate)
- **17 engineered features** including power, RUL, torque-wear interaction
- **5 failure types** classified: TWF, HDF, PWF, OSF, RNF

---

## What It Does

### 1. SQL-Based Data Analysis
Extracts failure patterns using SQLite queries — overall failure rate, failure by machine type, average sensor readings for failed vs normal machines, and high-risk machine identification.

### 2. Exploratory Data Analysis
- Failure distribution and rate by machine type (L/M/H)
- Boxplot comparison of sensor readings — normal vs failure
- Sensor correlation heatmap

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `power` | Torque × (2π × RPM / 60) |
| `RUL` | Remaining Useful Life (max wear − current wear) |
| `torque_wear` | Torque × Tool wear — direct OSF indicator |
| `temp_delta` | Process temp − Air temp |
| `wear_rate` | Tool wear / RPM |
| `rpm_torque` | RPM × Torque interaction |
| `rolling_temp_5` | 5-row rolling average of air temperature |
| `rolling_torque_10` | 10-row rolling average of torque |
| `lag_torque_1` / `lag_rpm_1` | Previous reading lag features |

### 4. ML Models
Both models trained with 5-fold cross-validation and GridSearchCV:

| Model | Train F1 | Test F1 |
|---|---|---|
| XGBoost | 0.9963 | **0.8358** |
| Random Forest | 0.8700 | 0.7516 |

XGBoost uses `scale_pos_weight`, L1/L2 regularization, subsample=0.8 to handle class imbalance.

### 5. Threshold Optimization
Default threshold (0.5) adjusted to **0.35** — balances recall ≥ 85% with precision ≥ 82% for practical maintenance use.

### 6. Real-time Monitoring Simulation
Streams sensor readings row by row, runs live inference, and triggers alerts:

| Failure Probability | Alert Level | Action |
|---|---|---|
| ≥ 0.70 | HIGH | Immediate maintenance |
| 0.40 – 0.70 | MEDIUM | Schedule within 7 days |
| < 0.40 | LOW | Continue monitoring |

### 7. Maintenance Scheduling
Generates a priority-based maintenance calendar for at-risk machines with probability scores and recommended action window (1 / 7 / 30 days).

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) — 10,000 rows of synthetic industrial sensor data from UCI Machine Learning Repository.

| Column | Description |
|---|---|
| Air / Process Temperature | Sensor readings in Kelvin |
| Rotational Speed | RPM of the machine |
| Torque | Applied torque in Nm |
| Tool Wear | Accumulated tool wear in minutes |
| Machine Failure | Binary target variable (0/1) |
| TWF / HDF / PWF / OSF / RNF | Individual failure type labels |

---

## Project Structure

```
Predictive_Maintenance_System/
├── data/
│   └── ai4i2020.csv
├── model/
│   └── maintenance_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── sensor_trends.png
│   ├── failure_distribution.png
│   ├── correlation_heatmap.png
│   └── maintenance_schedule.png
├── predictive_maintenance.ipynb
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## Tech Stack

`Python 3.13` `XGBoost` `scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `SQLite3` `joblib` `Jupyter Notebook`

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook predictive_maintenance.ipynb
```

Run all cells in order — model saves to `model/` and all plots save to `outputs/`.
