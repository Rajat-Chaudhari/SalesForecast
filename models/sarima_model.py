import pandas as pd
import mlflow
import datetime
import traceback
from statsmodels.tsa.statespace.sarimax import SARIMAX
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations

def build_sarima(train_data, target_col, model_config):
    """Build and fit SARIMA model."""
    order = tuple([model_config['p'],model_config['d'],model_config['q']])
    
    seasonal_order = tuple([model_config['P'],model_config['D'],model_config['Q'],model_config['m']])
    
    model = SARIMAX(train_data[target_col], order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    
    return fitted_model

def run_sarima(model_name,train_data, test_data, target_col, params):
    """Run SARIMA model."""
    
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
            
                fitted_model = build_sarima(train_data, target_col, params)
                forecasted_values = fitted_model.forecast(steps=len(test_data))
                
                save_performance_metrics(train_data[target_col], test_data[target_col], 
                                        fitted_model.fittedvalues, forecasted_values,
                                        model_name,params,run_id)
                
                save_model(fitted_model,model_name,run_id)
                
    except Exception as e:
        print("Error ::",traceback.format_exc())
        raise e


def predict_test():
    
    return None

def predict_train():
    
    
    return None

def forecast_future():
    
    return None
