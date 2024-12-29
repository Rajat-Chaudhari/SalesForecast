import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor

# Load dataset (example)
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model_name, report

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Classifier': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Use ThreadPoolExecutor to train models in parallel
with ThreadPoolExecutor(max_workers=len(models)) as executor:
    futures = {executor.submit(train_and_evaluate_model, model, name): name for name, model in models.items()}
    
    for future in futures:
        model_name, report = future.result()
        print(f"\nClassification report for {model_name}:\n")
        print(report)

