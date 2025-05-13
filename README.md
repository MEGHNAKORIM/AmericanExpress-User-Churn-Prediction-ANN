# 📌 AmericanExpress-User-Churn-Prediction-ANN

## 🧠 Overview
This project demonstrates an end-to-end **Artificial Neural Network (ANN)** model to predict whether a customer will exit (churn) based on the **American Express dataset**. It includes complete data preprocessing, model training, evaluation, and a conceptual **3D visualization** of the pipeline.

---

## 📂 Project Structure
├── data/
│ └── AmericanExpress Data Analysis for User Exit Prediction.csv
├── churn_prediction_ann.py
├── README.md
└── requirements.txt

---

## 🚀 Features
- Label and One-Hot Encoding for categorical features
- Feature Scaling using `StandardScaler`
- Model built with `TensorFlow` Keras ANN:
  - Input layer
  - 2 Hidden layers (ReLU activation)
  - Output layer (Sigmoid activation)
- Evaluation using accuracy and confusion matrix
- Single customer prediction

---

## 🛠️ Tech Stack
- Python 3.x
- Pandas, NumPy, Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## 📈 Model Performance
- **Accuracy:** ~XX.XX% *(Based on your test results)*
- **Confusion Matrix:** Displays TP, FP, FN, TN

---

## 📊 3D Visualization
The `3d_visualization.py` script shows a conceptual 3D pipeline:
- From raw data → preprocessing → ANN layers → prediction
- Useful for presentations and project demonstrations

---

## 🧪 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AmericanExpress-User-Churn-Prediction-ANN.git
   cd AmericanExpress-User-Churn-Prediction-ANN
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run model training
   ```bash
   python churn_prediction_ann.py

