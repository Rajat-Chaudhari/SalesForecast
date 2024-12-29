import pandas as pd
import numpy as np
import datetime
import mlflow
import traceback
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations

def build_lr(train_data, target_col, model_config):
    """Build and fit the Linear Regression model."""
    # Extract features and target
    X = train_data.drop(columns=[target_col])
    y = train_data[target_col]
    # Split train and test data
    # Use the first 80% of the data for training and the rest for testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize and fit the Linear Regression model
    model = LinearRegression(fit_intercept=model_config['fit_intercept'])
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def run_lr(model_name,train_data, test_data, target_col, params):
    """Run linear regression model."""
    try:
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_name = f'{model_name}_{timestamp}'
        mlflow.set_experiment(f'{experiment_name}')
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f'Experiment :{experiment_name}')
        
        mlflow.sklearn.autolog(silent=True,log_post_training_metrics=False)
        
        combinations = generate_param_combinations(params)
        
        for params in combinations:
            
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                run_id = run.info.run_id
            
                print(f'model name : {model_name} train data size : {train_data.shape} || target_col : {target_col} || params : {params} || run id : {run_id}')
             
                fitted_model,X_train, X_test, y_train, y_test = build_lr(train_data, target_col, params)
                
                prediction_train_lst = predict_train(fitted_model, X_train)
                prediction_test_lst = predict_test(fitted_model,X_test)
                
                actual_train_lst = y_train.tolist()
                actual_test_lst = y_test.tolist()
                
                save_performance_metrics(actual_train_lst, actual_test_lst,
                                        prediction_train_lst, prediction_test_lst,
                                        model_name,params,run_id)
                
                save_model(fitted_model,model_name,run_id)
                
    except Exception as e:
        print("Error ::",traceback.format_exc())
        raise e


def predict_train(model, X_train):
    """Predict on training data."""
    return model.predict(X_train)

def predict_test(model, X_test):
    """Predict on test data."""
    return model.predict(X_test)

def forecast_future(model, last_data, periods):
    """Forecast future values."""
    future_predictions = []
    input_data = last_data[-1].reshape(1, -1)

    for _ in range(periods):
        prediction = model.predict(input_data)[0]
        future_predictions.append(prediction)

        # For simplicity, shift the input with the prediction
        input_data = np.append(input_data[:, 1:], prediction).reshape(1, -1)

    return future_predictions

# def calculate_metrics(y_true, y_pred):
#     """Calculate MAE, MSE, and RMSE."""
#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     return mae, mse, rmse

# # Example usage
# if __name__ == "__main__":
#     # Load and preprocess data
#     df = pd.read_csv('AOD_RS_Actual.csv')
#     df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
#     df['Year'] = df['Month'].dt.year
#     df['Month_num'] = df['Month'].dt.month
#     df.drop(columns=['Month'], inplace=True)
#     df.fillna(df.mean(), inplace=True)

    # Define model configuration
    # model_config = {
    #     'linear': {
    #         'fit_intercept': True,
    #         'scale_features': True  # Use feature scaling instead of 'normalize'
    #     },
    #     'test_size': 0.2,
    #     'random_state': 42
    # }

#     # Build and fit the model
#     model, X_train, X_test, y_train, y_test = build_linear_regressor(df, 'RS_Actual', model_config)

#     # Get train and test predictions
#     train_forecast = predict_train(model, X_train)
#     test_forecast = predict_test(model, X_test)

#     # Calculate performance metrics
#     train_mae, train_mse, train_rmse = calculate_metrics(y_train, train_forecast)
#     test_mae, test_mse, test_rmse = calculate_metrics(y_test, test_forecast)

#     print("Train Metrics: MAE =", train_mae, "MSE =", train_mse, "RMSE =", train_rmse)
#     print("Test Metrics: MAE =", test_mae, "MSE =", test_mse, "RMSE =", test_rmse)

#     # Forecast future values for the next 6 months
#     future_forecast = forecast_future(model, X_test, periods=6)

#     print("\nFuture Forecast:")
#     print(future_forecast)
