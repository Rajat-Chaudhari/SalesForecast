import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
import joblib
import datetime
import mlflow

def save_results(model_name, train_data, test_data, model_results, forecasted_dates, forecasted_values):
    """Save training and testing results to an Excel file."""
    
    result_file = os.path.join('results', f"{model_name}_results.xlsx")
    print("result_file",result_file)
    print(test_data)
    
    with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
        
        train_predictions = model_results.fittedvalues
        train_results = pd.DataFrame({'Actual': train_data, 'Predicted': train_predictions})
        print("Train")
        print(train_results)
        
        train_results.to_excel(writer, sheet_name='Training Results')

        test_predictions = forecasted_values
        test_results = pd.DataFrame({'Actual': test_data, 'Predicted': test_predictions})
        print("Test")
        print(test_results)
        
        test_results.to_excel(writer, sheet_name='Testing Results')

        forecasted_results = pd.DataFrame({'Date': forecasted_dates, 'Forecasted': test_predictions})
        print("forecasted_results")
        print(forecasted_results)
        forecasted_results.to_excel(writer, sheet_name='Forecasted Results', index=False)
        
def mape_loss(predictions, targets):
    epsilon = 1e-7  # Small constant to avoid division by zero
    # Calculate MAPE and return the average score
    mape_values = [abs((t - p) / (t + epsilon))  for p, t in zip(predictions, targets)]
    return sum(mape_values) / len(mape_values)

def save_performance_metrics(train_data, test_data, train_predictions, test_predictions, model_name,params,run_id):
    """Calculate and save performance metrics for training and testing datasets."""
    
    train_mae = mean_absolute_error(train_data, train_predictions)
    train_mse = mean_squared_error(train_data, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mape = mape_loss(train_data, train_predictions)
    
    test_mae = mean_absolute_error(test_data, test_predictions)
    test_mse = mean_squared_error(test_data, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mape = mape_loss(test_data, test_predictions)

    metrics = {
        "Model": model_name,
        
        "Train MAPE":np.round(train_mape,decimals=3).item(),
        "Test MAPE":np.round(test_mape,decimals=3).item(),
        
        "Train MAE": round(train_mae,3),
        "Test MAE": round(test_mae,3),
        
        "Train MSE": round(train_mse,3),
        "Test MSE": round(test_mse,3),
        
        "Train RMSE": round(train_rmse,3),
        "Test RMSE": round(test_rmse,3),
        
        "parameters":str(params),
        "run_id" :run_id
    }
    
    ### Log Data Into Mlflow
    keys_to_exclude = ["Model", "parameters","run_id"]
    filtered_metrics = {k: v for k, v in metrics.items() if k not in keys_to_exclude}
    # Log the metrics
    print("metrics")
    print(filtered_metrics)
    mlflow.log_metrics(filtered_metrics)
    mlflow.log_params(params)

    metrics_df = pd.DataFrame([metrics])
    
    # Write Excel In Append Model
    file_path = f"results/{model_name}_performance_metrics.xlsx"
    try:
        # Load existing data from file
        existing_data = pd.read_excel(file_path)
        # Append new data
        updated_data = pd.concat([existing_data, metrics_df], ignore_index=True)
        # Write back to Excel
        updated_data.to_excel(file_path, index=False,sheet_name='Performance Metrics')

    except FileNotFoundError:
        # If file doesn't exist, create a new one with new_data only
        metrics_df.to_excel(file_path, index=False, sheet_name='Performance Metrics')
    
    
def save_model(model_results, model_name,run_id):
    """Save the trained model using joblib."""
    joblib.dump(model_results, os.path.join('tmp', f"{model_name}_model_{run_id}.joblib"))
