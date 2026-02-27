# =====================================
# 1️⃣ Import Required Libraries
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# =====================================
# 2️⃣ Load Dataset
# =====================================

data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print("\nClass Distribution:\n")
print(data['Class'].value_counts())

# =====================================
# 3️⃣ Feature Correlation Heatmap
# =====================================

plt.figure(figsize=(12,8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# =====================================
# 4️⃣ Prepare Features & Target
# =====================================

X = data.drop('Class', axis=1)
y = data['Class']

# Remove NaN rows if any
nan_indices = y[y.isna()].index
X = X.drop(nan_indices)
y = y.drop(nan_indices)

# =====================================
# 5️⃣ Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# 6️⃣ Logistic Regression Model
# =====================================

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# =====================================
# 7️⃣ Predictions
# =====================================

y_pred = model.predict(X_test)

# =====================================
# 8️⃣ Evaluation Metrics
# =====================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =====================================
# 9️⃣ Confusion Matrix (Heatmap)
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================================
# 🔟 ROC Curve
# =====================================

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
