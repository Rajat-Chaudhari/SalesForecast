import pandas as pd
from prophet import Prophet
import datetime
import mlflow
import traceback
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations


def build_prophet(train_data, target_col, time_col, model_config):
    """Build and fit Prophet model.""" 
    train_data = train_data.reset_index().rename(columns={time_col: 'ds', target_col: 'y'})
    
    model = Prophet(seasonality_mode=model_config['seasonality_mode'])
    model.fit(train_data)
    return model

def run_prophet(model_name,train_data, test_data, target_col, date_col, params, model_type):
    """Run Prophet model."""
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
               
                model_results = build_prophet(train_data, target_col, date_col, params)
                
                future = model_results.make_future_dataframe(periods=len(test_data))
                forecast = model_results.predict(future)
                forecasted_values = forecast['yhat'][-len(test_data):].values

                train_results = predict_data(model_results, train_data, date_col, target_col)
                test_results = predict_data(model_results, test_data, date_col, target_col)
                
                save_performance_metrics(train_data[target_col], test_data[target_col], 
                                         train_results.Prediction, test_results.Prediction,
                                         model_name,params,run_id)
                
                save_model(model_results,model_name,run_id)
                
    except Exception as e:
       print("Error ::",traceback.format_exc())

def predict_data(model,data,time_col,target_col):
    
    data = data.reset_index().rename(columns={time_col: 'ds', target_col: 'y'})
    
    forecast = model.predict(data[['ds']])
    actual_values = data['y'].values
    predicted_values = forecast['yhat'].astype(int).values[:len(actual_values)]
    data_df = data.copy()
    data_df['Prediction'] = predicted_values
    data_df
    
    return data_df

def forecast_future(model):
    
    future_dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='MS').to_frame(index=False, name='ds')
    forecast = model.predict(future_dates)
    forecast = forecast[['ds', 'yhat']]  
    
    return None
