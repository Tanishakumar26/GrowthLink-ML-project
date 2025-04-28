|# GrowthLink-ML-project
Credit Card Fraud Detection System
# Fraud Detection System

## 📋 Project Objective
The goal is to build a machine learning model that detects fraudulent transactions based on transaction attributes like amount, merchant information, user details, and timestamps.

The system focuses on:
- **Maximizing fraud detection accuracy**.
- **Minimizing false positives** (legitimate transactions wrongly flagged as fraud).
- **Providing explanations for model misclassifications**.

## 🛠️ How to Run This Project

1. **Clone this repository**
    ```bash
    git clone https://github.com/your_username/fraud_detection_project.git
    cd fraud_detection_project
    ```

2. **Install required libraries**
    ```bash
    pip install -r requirements.txt
    ```

3. **Place your dataset**  
   - Add `fraudTrain.csv` and `fraudTest.csv` inside the `data/` folder.

4. **Run the project**
    ```bash
    python main.py
    ```

The outputs (metrics, feature importance plots, confusion matrix) will be printed and visualized automatically.

---

## 📂 Project Structure
- `main.py`: Runs the full pipeline.
- `src/preprocessing.py`: Data cleaning and feature engineering.
- `src/model.py`: Model building and training.
- `src/evaluation.py`: Model evaluation and metrics.
- `src/explain_misclassifications.py`: Analyze false positives/negatives.
- `data/`: Contains your train/test datasets.
- `requirements.txt`: Lists required Python packages.
- `README.md`: This file.

---

## ⚙️ Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
