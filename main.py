import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess the data
# Convert diagnosis to numerical
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select features and target
X = df.drop(['id', 'diagnosis'], axis=1)
# After dropping 'id' and 'diagnosis', check if the last column is unnamed and drop it if it exists.
if 'Unnamed: 32' in X.columns:
    X = X.drop('Unnamed: 32', axis=1)

y = df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, recall, and ROC-AUC
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('roc_curve.png')
plt.close()

# Sigmoid function explanation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x_sigmoid = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x_sigmoid)
plt.figure(figsize=(8, 6))
plt.plot(x_sigmoid, y_sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid()
plt.savefig('sigmoid_function.png')
plt.close()


# Threshold tuning
y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
precision_scores, recall_scores, threshold_values = [], [], []

for threshold in np.arange(0.1, 1.0, 0.05):
    y_pred_tuned = (y_pred_proba_train > threshold).astype(int)
    precision_scores.append(precision_score(y_train, y_pred_tuned, zero_division=0))
    recall_scores.append(recall_score(y_train, y_pred_tuned))
    threshold_values.append(threshold)

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, precision_scores, label='Precision')
plt.plot(threshold_values, recall_scores, label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs. Threshold")
plt.legend()
plt.grid()
plt.savefig('precision_recall_threshold.png')
plt.close()