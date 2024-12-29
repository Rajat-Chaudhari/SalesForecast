import pandas as pd
import numpy as np
import traceback
import datetime
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations



def build_lstm( X_train, y_train, target_col, model_config):
    """Build and compile the LSTM model."""
    
    n_input = model_config['n_input']
    n_features = model_config['n_features']
    n_units = model_config['n_units']
    dropout_rate = model_config['dropout_rate']
    epochs = model_config['epochs']
    batch_size = model_config['batch_size']

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model

def run_lstm(model_name,train_data, test_data, target_col, params):
    """Run LSTM Univariate model."""
    
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
                
                n_input = params['n_input']
                X_train, y_train = prepare_data(train_data[target_col], n_input)
                fitted_model = build_lstm(X_train, y_train,target_col, params)
                
                prediction_train_lst = predict_train(fitted_model, X_train)
                test_data = pd.concat([train_data.tail(n_input),test_data])
                X_test, y_test = prepare_data(test_data[target_col], n_input)
                prediction_test_lst = predict_test(fitted_model,X_test)
                actual_train_lst = train_data[target_col].tolist()[n_input:]
                actual_test_lst = test_data[target_col].tolist()[n_input:]
                
                save_performance_metrics(actual_train_lst, actual_test_lst,
                                         prediction_train_lst, prediction_test_lst,
                                         model_name,params,run_id)
                save_model(fitted_model,model_name,run_id)
                
    except Exception as e:
        print("Error ::",traceback.format_exc())
        raise e
    
def prepare_data(data, n_input):
    """Prepare input and output data for LSTM."""
    try:
        print("Data......")
        scaled_data = data
        scaled_data.reset_index(inplace=True,drop=True)
        X, y = [], []
        for i in range(len(scaled_data) - n_input):
            X.append(scaled_data[i:i + n_input])
            y.append(scaled_data[i + n_input])

        X = np.array(X)
        y = np.array(y)
        return X, y
    
    except Exception as e:
        print(traceback.format_exc(e))

def predict_train(model, X_train):
    """Predict in-sample values."""
    print(X_train)
    train_forecast = model.predict(X_train)
    return train_forecast.flatten()
    

def predict_test(model, X_test):
    """Predict test values."""
    test_forecast = model.predict(X_test)
    return test_forecast.flatten()

def forecast_future(model, last_data, periods, n_input):
    """Forecast future values."""
    predictions = []
    input_data = last_data[-n_input:]

    for _ in range(periods):
        input_reshaped = input_data.reshape((1, n_input, 1))
        prediction = model.predict(input_reshaped)[0, 0]
        predictions.append(prediction)

        # Shift input data for the next prediction
        input_data = np.append(input_data[1:], prediction)

    return predictions

def inverse_scale(data, forecast):
    """Inverse MinMax scaling."""
    scaler = MinMaxScaler().fit(data.values.reshape(-1, 1))
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1))


