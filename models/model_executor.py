from models.arima_model import run_arima
from models.sarima_model import run_sarima
from models.prophet_model import run_prophet
from models.ets_model import run_ets
from models.ar_model import run_ar
from models.tbats_model import run_tbats
from models.lstm_model import run_lstm
from models.linear_regression_model import run_lr
from models.randomforest_regression_model import run_rf
from models.elasticnet_regression_model import run_enr
from models.catboost_model import run_catboost
from models.lightgbm_model import run_lightgbm
from models.xgboost_model import run_xgboost
from models.svr_model import run_svr
from models.full_connected_nn import run_fully_connected_nn


def prepare_models_for_execution(model_names, model_params, train_data, test_data, 
                                 input_features_lst,target_col,date_col, model_type):
    """
    Prepares a list of model functions to be executed.
    
    Args:
        model_names (list): List of model names to run (e.g., ['ARIMA', 'Prophet']).
        model_params (dict): Corresponding parameters for each model.
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        target_col (str): Target column for prediction.
        date_col (str): Date column for time series.
        model_type (str): Either 'univariate' or 'multivariate'.
        
    Returns:
        List of functions ready to be executed in threading.
    """
    try:
        
        model_functions = []

        for model_name in model_names:
            params = model_params.get(model_name, {})  # Get parameters specific to the model
            
            if model_name == 'ARIMA':
                model_functions.append(lambda model_name=model_name, params=params: run_arima(model_name,train_data, test_data, target_col, params))
            elif model_name == 'SARIMA':
                model_functions.append(lambda model_name=model_name, params=params: run_sarima(model_name,train_data, test_data, target_col, params))
            elif model_name == 'Univariate-Prophet':
                model_functions.append(lambda model_name=model_name, params=params: run_prophet(model_name,train_data, test_data, target_col, date_col, params,model_type))
            elif model_name == 'ETS':
                model_functions.append(lambda model_name=model_name, params=params: run_ets(model_name,train_data, test_data, target_col, params))
            elif model_name == 'AR':
                model_functions.append(lambda model_name=model_name, params=params: run_ar(model_name,train_data, test_data, target_col, params))
            elif model_name == 'Univariate-TBATS':
                model_functions.append(lambda model_name=model_name, params=params: run_tbats(model_name,train_data, test_data, target_col, params))
            elif model_name == 'Univariate-LSTM':
                model_functions.append(lambda model_name=model_name, params=params: run_lstm(model_name,train_data, test_data, target_col, params))
            elif model_name == 'LinearRegressor':
                model_functions.append(lambda model_name=model_name, params=params: run_lr(model_name,train_data, test_data, target_col, params))
            elif model_name == 'ElasticNet':
                model_functions.append(lambda model_name=model_name, params=params: run_enr(model_name,train_data, test_data, target_col, params))
            elif model_name == 'RandomForest':
                model_functions.append(lambda model_name=model_name, params=params: run_rf(model_name,train_data, test_data, target_col, params))
            elif model_name == 'XGBoost':
                model_functions.append(lambda model_name=model_name, params=params: run_xgboost(model_name,train_data, test_data, target_col, params))
            elif model_name == 'LightGBM':
                model_functions.append(lambda model_name=model_name, params=params: run_lightgbm(model_name,train_data, test_data, target_col, params))
            elif model_name == 'CatBoost':
                model_functions.append(lambda model_name=model_name, params=params: run_catboost(model_name,train_data, test_data, target_col, params))
            elif model_name == 'SVR':
                model_functions.append(lambda model_name=model_name, params=params: run_svr(model_name,train_data, test_data, target_col, params))
            elif model_name == 'FullyConnectedNN':
                model_functions.append(lambda model_name=model_name, params=params: run_fully_connected_nn(model_name,train_data, test_data, target_col, params))
            
            
        return model_functions,0
    
    except Exception as e:
        return None , 6
        

