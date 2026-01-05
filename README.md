# 🛡️ FraudGuard: Real-Time Anomaly Detection

## 📌 Project Overview
This project implements an **Unsupervised Learning** engine using **Isolation Forest** to detect anomalies in real-time without needing labeled training data.

**Key Feature:** Converts raw anomaly scores into a human-readable **0-100 Risk Score** for bank analysts.

## 🔧 Tech Stack
* **Core:** Python 3.9, Pandas, NumPy
* **ML:** Isolation Forest (Unsupervised), RobustScaler (Outlier-safe scaling)
* **Visualization:** Matplotlib, Seaborn
* **App:** Streamlit (Simulation Dashboard)

## 📊 Key Results
| Metric | Result | Context |
|:-------|:-------|:--------|
| **Fraud Caught (Recall)** | ~31% | Detected without *any* labels |
| **Legit Risk Score** | **~13 / 100** | Low false alarm pressure |
| **Fraud Risk Score** | **~60 / 100** | High separation signals |

## 🚀 How to Run Locally
1.  Clone the repo.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Dashboard:
    ```bash
    streamlit run src/app.py
    ```

## 📂 Project Structure
* `src/app.py`: Real-time transaction simulator.
* `notebooks/`:
    * `01_exploration.ipynb`: Visualization of the 0.17% imbalance.
    * `02_preprocessing.ipynb`: Robust Scaling for financial outliers.
    * `03_modeling.ipynb`: Training Isolation Forest and calibrating Risk Scores.
* `models/`: Serialized models (Joblib).
