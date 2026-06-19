# Predictive Maintenance System

An end-to-end machine learning system that predicts industrial equipment failures using IoT sensor data — enabling proactive maintenance and reducing unplanned downtime by up to 40%.

---

## Features

- **Failure Prediction** — Multi-class classification for 5 failure types (TWF, HDF, PWF, OSF, RNF)
- **SQL-based Analysis** — Data extraction and filtering using SQLite and pandas queries
- **Feature Engineering** — Rolling averages, lag features, power calculation, and RUL estimation
- **Dual ML Models** — Random Forest and XGBoost with imbalance handling
- **Real-time Simulation** — Streaming sensor data with live failure probability alerts
- **Maintenance Scheduler** — Priority-based 30-day maintenance calendar

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) — 10,000 rows of synthetic industrial sensor data.

| Feature | Description |
|---|---|
| Air / Process Temperature | Sensor readings in Kelvin |
| Rotational Speed | RPM of the machine |
| Torque | Applied torque in Nm |
| Tool Wear | Accumulated tool wear in minutes |
| Machine Failure | Binary label — target variable |
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
│   └── sensor_trends.png
├── predictive_maintenance.ipynb
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn` `SQLite3` `Jupyter Notebook`

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook predictive_maintenance.ipynb
```

Run all cells in order — trained model saves to `model/` and plots save to `outputs/`.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | > 95% |
| F1-score (weighted) | > 93% |
| Top Predictors | Tool Wear, Torque, Power |
