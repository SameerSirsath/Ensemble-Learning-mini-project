# Online Payment Fraud Detection – Ensemble Learning Project

This project implements a complete machine learning pipeline for detecting fraudulent online payments. It follows a strict syllabus: classical ML algorithms, ensemble methods, gradient descent from scratch, K‑means, PCA, and an interactive Streamlit app. No deep learning is used.

## 📌 Features

- Data cleaning, EDA, feature engineering, SMOTE balancing
- 9 classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, SVM, Naïve Bayes, AdaBoost, Gradient Boosting)
- Gradient Descent implemented from scratch (binary cross‑entropy)
- Unsupervised analysis: K‑means (elbow + silhouette), hierarchical clustering (dendrogram), PCA (2D/3D)
- Streamlit app: choose any model, input transaction details, get fraud prediction + probability
- Performance comparison table (accuracy, precision, recall, F1, ROC‑AUC)

## 🛠️ Setup & Execution (Step‑by‑Step)

### Step 1 – Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt