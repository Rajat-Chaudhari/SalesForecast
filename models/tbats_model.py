import pandas as pd
from sktime.forecasting.tbats import TBATS
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations
import numpy as np
import traceback
import datetime
import mlflow

def build_tbats(train_data,target_col,model_config):
    """Build and fit the TBATS model."""

    forecaster = TBATS(
        use_box_cox=model_config['use_box_cox'],
        use_trend=model_config['use_trend'],
        use_damped_trend=model_config['use_damped_trend'],
        sp=model_config['sp'],
        use_arma_errors=model_config['use_arma_errors'],
        n_jobs=model_config['n_jobs']
    )
    forecaster.fit(train_data[target_col])
  
    return forecaster

def run_tbats(model_name,train_data, test_data, target_col, params):
    """Run TBATS model."""
    try:
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_name = f'{model_name}_{timestamp}'
        mlflow.set_experiment(f'{experiment_name}')
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f'Experiment :{experiment_name}')
    
        mlflow.autolog(silent=True)
        
        combinations = generate_param_combinations(params)
    
        for params in combinations:
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                run_id = run.info.run_id
                print(f'model name : {model_name} train data size : {train_data.shape} || target_col : {target_col} || params : {params} || run id : {run_id}')
                fitted_model = build_tbats(train_data, target_col, params)
                
                train_predictions = predict_train(fitted_model, train_data)
                forecasted_values = predict_test(fitted_model,test_data)
                
                save_performance_metrics(train_data[target_col], test_data[target_col], 
                                         train_predictions, forecasted_values,
                                         model_name,params,run_id)
                
                save_model(fitted_model,model_name,run_id)
                
    except Exception as e:
        print("Error ::",traceback.format_exc())
        raise e

def predict_train(forecaster, train_data):
    """Get in-sample predictions (train forecast)."""
    fh = np.arange(1, len(train_data) + 1)  # Forecasting horizon for the entire training set
    train_forecast = forecaster.predict(fh=fh)
    return train_forecast

def predict_test(forecaster, test_data):
    """Get out-of-sample predictions (test forecast)."""
    fh = np.arange(1, len(test_data) + 1)  # Forecasting horizon for the test set
    test_forecast = forecaster.predict(fh=fh)
    return test_forecast

def forecast_future(forecaster, periods):
    """Forecast future values beyond the test period."""
    fh = np.arange(1, periods + 1)  # Forecasting horizon for future periods
    future_forecast = forecaster.predict(fh=fh)
    return future_forecast

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, MSE, and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# # Example usage
# if __name__ == "__main__":
#     # Load data
#     df = pd.read_csv('AOD_RS_OFFICIAL_DATA.csv')
#     df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
#     df.set_index('Date', inplace=True)

#     # Split train and test data
#     train_data = df['Sales'][:-6]  # All data except the last 6 months for training
#     test_data = df['Sales'][-6:]   # Last 6 months for testing

#     # Model configuration
#     model_config = {
        # 'tbats': {
        #     'use_box_cox': False,
        #     'use_trend': False,
        #     'use_damped_trend': False,
        #     'sp': 12, 
        #     'use_arma_errors': False,
        #     'n_jobs': 1  
        # }
#     }

#     # Build and fit the TBATS model
#     forecaster = build_tbats(train_data, model_config)

#     # Get train and test predictions
#     train_forecast = predict_train(forecaster, train_data)
#     test_forecast = predict_test(forecaster, test_data)

#     # Forecast future sales for the next 6 months
#     future_forecast = forecast_future(forecaster, periods=6)

#     # Calculate and print performance metrics
#     train_mae, train_mse, train_rmse = calculate_metrics(train_data, train_forecast)
#     test_mae, test_mse, test_rmse = calculate_metrics(test_data, test_forecast)

#     print("Train Metrics: MAE =", train_mae, "MSE =", train_mse, "RMSE =", train_rmse)
#     print("Test Metrics: MAE =", test_mae, "MSE =", test_mse, "RMSE =", test_rmse)

#     print("\nFuture Forecast:")
#     print(future_forecast)
