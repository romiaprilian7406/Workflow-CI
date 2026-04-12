import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

mlflow.set_experiment("Heart-Disease-Classification")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(BASE_DIR, "heart_disease_train.csv"))
test_df  = pd.read_csv(os.path.join(BASE_DIR, "heart_disease_test.csv"))

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test  = test_df.drop("target", axis=1)
y_test  = test_df["target"]

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

mlflow.sklearn.autolog()

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print("Model berhasil dilatih!")
