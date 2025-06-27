# 📧 Spam Email Classifier using Machine Learning

A machine learning project that classifies emails as **spam** or **not spam** using the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase). It demonstrates a complete ML pipeline from data preprocessing, feature selection, model training, to evaluation and (optional) deployment.

---

## 🧠 Problem Statement

Spam emails are a common issue, and manually filtering them is inefficient. This project builds a classifier that can automatically detect spam emails based on their content and metadata.

---

## 📂 Dataset

- **Name:** Spambase Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase)
- **Instances:** 4601
- **Attributes:** 57 features + 1 label (`1 = spam`, `0 = not spam`)
- **Format:** `.data` CSV file

---

## 💡 ML Pipeline / Logic Used

1. **Data Loading & Exploration**
   - Load CSV using pandas
   - Understand features and target distribution

2. **Preprocessing**
   - Handle missing values
   - Normalize features
   - Feature selection (optional: PCA)

3. **Model Building**
   - Algorithms used:
     - Logistic Regression
     - Naive Bayes (recommended for text data)
     - Random Forest
     - SVM (optional)
   - Trained on 80% of data, tested on 20%

4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC score

5. **(Optional) Deployment**
   - Export model using `joblib`
   - Build simple `Streamlit` app for user testing

---

## 📊 Model Metrics

| Model              | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Naive Bayes        |  **~89%** | 85%       | 92%    | 88%      |
| Logistic Regression| ~86%     | ...       | ...    | ...      |
| Random Forest      | ~91%     | ...       | ...    | ...      |

*(Exact results may vary)*

---

## 🧰 Tools & Libraries Used

- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- Git, GitHub
- (Optional) Streamlit

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/the-aditya-work/spam-email-classifier.git
   cd spam-email-classifier

