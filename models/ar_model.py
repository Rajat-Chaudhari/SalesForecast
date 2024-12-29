import pandas as pd
import traceback
import datetime
import mlflow
from statsmodels.tsa.ar_model import AutoReg
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations


def build_ar(train_data, target_col, model_config):
    """Build and fit AR model."""

    lags = model_config['lags']  # Number of lags to use
    model = AutoReg(train_data[target_col], lags=lags)
    model_results = model.fit()
    return model_results,0
    
def run_ar(model_name,train_data, test_data, target_col, params):
    """Run ARIMA model."""
    
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
                
                fitted_model,status = build_ar(train_data, target_col, params)
    
                forecasted_values = fitted_model.forecast(steps=len(test_data))
                train_prediction = fitted_model.fittedvalues.tolist()
                test_prediction = forecasted_values.tolist()
                
                train_data_lst=train_data[target_col].tolist()
                train_data_lst = train_data_lst[params['lags']:]
                
                save_performance_metrics(train_data_lst, test_data[target_col], 
                                        train_prediction, test_prediction,
                                        model_name,params,run_id)
                
                save_model(fitted_model,model_name,run_id)
                    
    except Exception as e:
        print("Error :: ",traceback.format_exc())
        raise e
                        
def predict_train(model_results, train_data):
    """Get in-sample predictions (train forecast)."""
    train_forecast = model_results.fittedvalues
    return train_forecast

def predict_test(model_results, test_data):
    """Get out-of-sample predictions (test forecast)."""
    steps = len(test_data)
    test_forecast = model_results.forecast(steps=steps)
    return test_forecast

def forecast_future(model_results, periods):
    """Forecast future values beyond the test period."""
    future_forecast = model_results.forecast(steps=periods)
    return future_forecast

