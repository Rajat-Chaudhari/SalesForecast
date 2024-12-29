import pandas as pd
import os
import json
import re

# Function to handle dynamic file loading
def load_data(file_path, sheet_name , date_col , target_col):
    """
    Load data from a file, checking the file extension to determine the format.

    Args:
        file_path (str): The path to the file (CSV or Excel).
        sheet_name (str, optional): The name of the sheet to read from Excel. Defaults to None, which reads the first sheet.
        exception_file (str): Path to the exceptions.json file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        
        status,data = 0 , None
        # Check if the file exists
        if not os.path.isfile(file_path):
            return data,1

        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.csv':
            # Read CSV file
            data = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xls', '.xlsx']:
            # Read Excel file, default to the first sheet if no sheet_name is provided
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return data,2
        
        if date_col not in data.columns:
            return data,3
        if target_col not in data.columns:
            return data,4
        if len(data) <= 6:
            return data,5
        
        data['Date'] = data['Date'].str.strip()
        data['Date'] = pd.to_datetime(data['Date'], format='%b-%y') 
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        return data,status
        
    except Exception as e:
        return data,6
    
def split_data(data, train_size):
    """Split the data into training and testing datasets."""
    try:
        train, test = data[:train_size], data[train_size:]
        return train,test,0
    
    except:
        return None, None ,6



def is_valid_string(input_str):
    """ Check if the string only contains alphabets, spaces, and valid special characters. """
    return bool(re.match(r"^[a-zA-Z-]+$", input_str))

def is_valid_number(input_value):
    """ Check if the input is a valid integer between 0 and 100. """
    return isinstance(input_value, int) and 0 <= input_value <= 100

def check_input_data(input_dict, exception_file="exceptions.json"):
   
    try:
        # Check if file path exists
        if not isinstance(input_dict.get('file_path'), str) or not input_dict['file_path']:
            return 1
        # Check if target_col is a valid string
        if not is_valid_string(input_dict.get('target_col', '')):
            return 2
        # Check if date_col is a valid string
        if not is_valid_string(input_dict.get('date_col', '')):
            return 3
        # Check model_type
        if input_dict.get('model_type') not in ['univariate', 'multivariate']:
            return 4
        # Check mode
        if input_dict.get('mode') not in ['manual', 'auto']:
            return 5
        # Check training_data_size is a valid integer
        if not is_valid_number(input_dict.get('training_data_size')):
            return 6
        # Check freq is a valid string
        if not isinstance(input_dict.get('freq'), str) or not is_valid_string(input_dict['freq']):
            return 7
        # Check selected_model_lst is a valid list of models
        selected_model_lst = input_dict.get('selected_model_lst', [])
        if not isinstance(selected_model_lst, list) or any(not is_valid_string(model) for model in selected_model_lst):
            return 8
        
        if input_dict.get('model_kpi') not in ['MAE','MAPE','MSE','RMSE']:
            return 10
        
        kpi_range = input_dict.get("kpi_range")
       
        # Parse the range from the string format "min - max"
        min_val, max_val = map(float, kpi_range.split('-'))
        # Validate that both min and max values are within 0.01 to 1.00
        if not (0.01 <= min_val <= 1.00 and 0.01 <= max_val <= 1.00):
            return 11
          

        return 0
        
    except Exception as e:
        return 9
    
    
    
def validate_and_confirm_models(user_models, repo_models):
    try:
        # Find models that are available in the repository
        available_models = [model for model in user_models if model in repo_models]
        # Find models that are not available in the repository
        unavailable_models = [model for model in user_models if model not in repo_models]
        if not available_models:
            return 7
        if unavailable_models:
            return 8
        
        return 0
    
    except Exception as e:
        
        return 6
    
    
# Function to retrieve parameters for a list of models
def get_params_for_models(model_list,model_dict):
    try:
        model_params_dict = {}
        for model_name in model_list:
            if model_name in model_dict['model_params']:
                model_params_dict[model_name] = model_dict['model_params'][model_name]
            else:
                model_params_dict[model_name] = "Model not found in the dictionary."
                
        return model_params_dict,0
    
    except Exception as e:
        return None,6
    
    
