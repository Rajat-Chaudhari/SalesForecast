import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Example: Load dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Logistic Regression': (LogisticRegression(), {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}),
    'Support Vector Classifier': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]})
}


best_models = {}

for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best model for {model_name}: {grid_search.best_estimator_}")
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification report for {model_name}:\n")
    print(classification_report(y_test, y_pred))
