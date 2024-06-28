pip install numpy pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample data (dataset)
# For demonstration, we'll use the Iris dataset and modify it to simulate ransomware detection
data = datasets.load_iris()
X = data.data
y = (data.target != 0).astype(int)  # Simulating a binary classification problem

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Behavioral Detection (BD) - Ensemble of SVMs
svm1 = SVC(probability=True, kernel='linear')
svm2 = SVC(probability=True, kernel='rbf')
svm3 = SVC(probability=True, kernel='poly')

ensemble_clf = VotingClassifier(estimators=[
    ('svm1', svm1), ('svm2', svm2), ('svm3', svm3)], voting='soft')

ensemble_clf.fit(X_train, y_train)

# Anomaly Detection (AD) - Single SVM
ad_clf = SVC(probability=True, kernel='sigmoid')
ad_clf.fit(X_train[y_train == 0], y_train[y_train == 0])  # Train only on benign data

# Confidence scores and dynamic weights
def calculate_confidence(model, X):
    return model.predict_proba(X)[:, 1]  # Confidence score for the positive class

bd_confidence = calculate_confidence(ensemble_clf, X_test)
ad_confidence = calculate_confidence(ad_clf, X_test)

# Assign dynamic weights based on confidence levels
bd_weight = bd_confidence / (bd_confidence + ad_confidence)
ad_weight = ad_confidence / (bd_confidence + ad_confidence)

# Decision Fusion
bd_decision = ensemble_clf.predict(X_test)
ad_decision = ad_clf.predict(X_test)

fused_decision = (bd_weight * bd_decision + ad_weight * ad_decision) >= 0.5
fused_decision = fused_decision.astype(int)

# Evaluation
accuracy = accuracy_score(y_test, fused_decision)
precision = precision_score(y_test, fused_decision)
recall = recall_score(y_test, fused_decision)
f1 = f1_score(y_test, fused_decision)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
