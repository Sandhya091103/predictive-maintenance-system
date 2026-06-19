# AI-Based Predictive Maintenance System

## Project Owner
- **Name:** Sandhya Singh
- **Email:** singhsandhya171@gmail.com
- **GitHub:** Sandhya091103

## Project Goal
Build an AI-based predictive maintenance system for industrial machinery using machine learning and IoT sensor data.

**Targets from resume:**
- Reduce machinery downtime by 40%
- Accurate failure prediction
- Optimized maintenance scheduling
- Real-time monitoring and forecasting
- SQL queries for data extraction, filtering, and analysis

## Dataset
- User will provide the dataset (IoT sensor data from industrial machinery)
- Expected columns: sensor readings (temperature, vibration, pressure, etc.), timestamps, machine ID, failure label
- Common public datasets: NASA CMAPSS (turbofan engine), AI4I 2020 Predictive Maintenance Dataset (UCI)
- User will place the data file in this folder before starting

## What Needs to Be Built
1. Data loading and SQL-based analysis (using SQLite or pandas SQL queries)
2. Exploratory Data Analysis (EDA) — sensor trends, failure patterns
3. Feature engineering — rolling averages, lag features, RUL (Remaining Useful Life)
4. ML models — Random Forest, XGBoost, or LSTM for failure prediction
5. Real-time monitoring simulation
6. Maintenance scheduling logic based on predictions
7. Evaluation — accuracy, precision, recall, F1, confusion matrix
8. Visualizations — sensor trends, failure prediction timeline

## Tech Stack
- Python, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
- SQLite / pandas SQL for data querying
- Jupyter Notebook

## Project Structure (to be created)
```
Predictive_Maintenance_System/
├── data/
│   └── sensor_data.csv        ← user will provide this
├── predictive_maintenance.ipynb
├── model/
│   └── maintenance_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── sensor_trends.png
├── requirements.txt
└── README.md
```

## Instructions for Claude
- User will provide the dataset — start coding once it's in the data/ folder
- Write clean, well-structured Jupyter Notebook
- Include SQL queries using pandas `.query()` or SQLite for data analysis section
- Do NOT add Co-Authored-By in any git commits (user preference)
- Git user is already configured as Sandhya Singh
- GitHub repo link will be provided by user — set it as remote and push when done
