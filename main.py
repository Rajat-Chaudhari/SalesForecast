import os
import pandas as pd
import traceback
import mlflow
from config import load_config_json,load_config
from data_loader import load_data, split_data , check_input_data , validate_and_confirm_models,get_params_for_models
from data_preprocessing import preprocess_data
from utils.threading_manager import run_models_in_batches , run_models_sequentially
from models.model_executor import prepare_models_for_execution


def main(model_dict):
    
    # User inputs (typically from UI, here hardcoded for demonstration)
    try:
        
        #current_dir = os.path.dirname(__file__)
        #config_path = os.path.join(current_dir, 'input.json')
        #input_config = load_config(config_path)
        
        input_config = model_dict
        
        print("input_config")
        print(input_config)
        
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, 'exception.json')
        exceptions = load_config_json(config_path)
        
        # Check Input Data
        status = check_input_data(input_config)
        if status != 0:
            return exceptions['invalid_input'][str(status)]
            
        file_path = input_config['file_path']
        sheet_name = None # input_config['sheet_name']
        target_col = input_config['target_col']
        date_col = input_config['date_col']
        model_type = input_config['model_type']
        model_type = model_type.strip().lower()
        mode = input_config['mode']
        mode = mode.strip().lower()
        training_data_size = input_config['training_data_size']
        freq = input_config['freq']
        
        print("Loading Data............")
        # Load data
        data,status = load_data(file_path,sheet_name,date_col,target_col)
        if status != 0 :
            return exceptions['file_not_found'][str(status)]

        print("Split Data............")
        # Split data into training and test sets
        train_data, test_data,status = split_data(data, input_config['training_data_size'])
        if status != 0 :
            return exceptions['file_not_found'][str(status)]
        
        # Fetch models and parameters from config.json
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, 'config.json')
        config = load_config_json(config_path)
        
        print(f'Model Type : {model_type} || Model Mode : {mode}')
        
        # Fetch models and parameters
        if mode == 'manual':
            
            # Get models and parameters from the user
            model_names = input_config['selected_model_lst']
            model_names = [model.strip() for model in model_names]
        
        else:
            model_names = config[f'{model_type}_models']
            model_names = [model.strip() for model in model_names]
            
            #model_params = config['model_params'] 
            
        status = validate_and_confirm_models(model_names, config[f'{model_type}_models'])
        if status != 0:
            return exceptions['file_not_found'][str(status)]
            
        model_params,status = get_params_for_models(model_names,config) 
        if status != 0:
            return exceptions['file_not_found'][str(status)]
         
            
        print(f'model_name :{model_names}')
        print(f'model_params :{model_params}')
        
        print(f'Run Model of Type : {model_type}')
        
        if model_type == 'multivariate':
            
            # preprocess_data(df, missing_strategy="mean", encoding_type="label", 
            #                 scaling_type="standard", columns_to_encode=None, fill_value=None)
            
            train_data['Month'] = pd.to_datetime(train_data['Date']).dt.month
            train_data.drop(columns=['Date'], inplace=True)
            
            test_data['Month'] = pd.to_datetime(test_data['Date']).dt.month
            test_data.drop(columns=['Date'], inplace=True)
        
        
        
        input_features_lst = None
        
        print("Models :",model_names)
        # Prepare the model execution pipeline
        
        model_functions,status = prepare_models_for_execution(model_names, model_params, 
                                                                train_data, test_data, 
                                                                input_features_lst,target_col, date_col, model_type)
        if status != 0:
            return exceptions['file_not_found'][str(status)]
        
        # Run models using threading (3 at a time)
        
        # print(f'threading started')
        # batch_size =3
        # run_models_in_batches(model_functions, batch_size)
        # print('threading ended')
        
        print(model_functions)
        status = run_models_sequentially(model_functions)
        
        if status != 0:
            return exceptions['file_not_found'][str(status)]
        
    except Exception as e:
        print(traceback.format_exc(e))
        return exceptions['file_not_found'][str(6)]
 
 
def start_model_training(model_dict):
    
    db_config = load_config()
    
    backend_store_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:5432/{db_config['database']}"
    default_artifact_root = "./artifacts" 
     
    mlflow.set_tracking_uri(backend_store_uri)
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    status = main(model_dict)
    print(status)
    
    
    
    
# Get Model Summar Details
    
def get_model_details(exp_name,exp_run_name,session_id):
     # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(exp_name)
    
    if not experiment:
        return "Experiment not found"

    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    print(runs)
    # Find the specific run by name
    run = runs[runs['tags.mlflow.runName'] == exp_run_name]
    
    if run.empty:
        return "Run not found"

    run_id = run.run_id.values[0]  # Get the run ID
    mlflow_run = mlflow.get_run(run_id)

    # Extract the relevant information
    run_info = mlflow_run.info
    run_data = mlflow_run.data

    # Prepare the response
    response = {
        "run_id": run_info.run_id,
        "run_name": run_info.run_name,
        "experiment_id": run_info.experiment_id,
        "status": run_info.status,
        "start_time": run_info.start_time,
        "end_time": run_info.end_time,
        "artifact_uri": run_info.artifact_uri,
        "metrics": run_data.metrics,
        "parameters": run_data.params,
        "tags": run_data.tags,
        "session_id":session_id
    }
    
    return response

# def get_data_insights(file, filename, session_id):
#     response_json = process_file(file, filename, session_id)
#     return response_json
               
# if __name__ == "__main__":
    
