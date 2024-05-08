import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Dataset.csv').drop_duplicates()

X = data.drop('Diabetes_012', axis=1)
y = data['Diabetes_012']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_tree = DecisionTreeClassifier()

model_tree.fit(X_train_scaled, y_train)
train_predictions = model_tree.predict(X_train_scaled)
test_predictions = model_tree.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Decision Tree without Oversampling")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print(classification_report(y_test, test_predictions))
print("ROC-AUC:", roc_auc_score(y_test, model_tree.predict_proba(X_test_scaled), multi_class='ovr'))

#with oversampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
model_tree.fit(X_train_smote, y_train_smote)
train_predictions_smote = model_tree.predict(X_train_smote)
test_predictions_smote = model_tree.predict(X_test_scaled)
train_accuracy_smote = accuracy_score(y_train_smote, train_predictions_smote)
test_accuracy_smote = accuracy_score(y_test, test_predictions_smote)

print("Decision Tree after Oversampling")
print("Train Accuracy:", train_accuracy_smote)
print("Test Accuracy:", test_accuracy_smote)
print(classification_report(y_test, test_predictions_smote))
print("ROC-AUC:", roc_auc_score(y_test, model_tree.predict_proba(X_test_scaled), multi_class='ovr'))
