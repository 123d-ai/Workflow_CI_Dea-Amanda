import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Membaca data_clean.csv...")
df = pd.read_csv("data_clean.csv")

X = df.drop(columns=['default'])
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Melatih model untuk CI Pipeline...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")
    print("Model berhasil dilatih dan disimpan oleh MLflow!")
