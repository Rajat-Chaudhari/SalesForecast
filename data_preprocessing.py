import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Exception handling utility (assuming exceptions.json has relevant messages)
def get_exception_message(exception_code):
    exceptions = {
        "missing_value_handling_failed": "Error handling missing values.",
        "encoding_failed": "Error during encoding.",
        "scaling_failed": "Error during scaling.",
        "preprocessing_failed": "Error during data preprocessing."
    }
    return exceptions.get(exception_code, "An unknown error occurred.")

# Function to handle missing values
def handle_missing_values(df, strategy="mean", fill_value=None):
    """
    Handle missing values in the DataFrame.
    Args:
        df (pd.DataFrame): The input dataframe.
        strategy (str): The strategy to use for imputing missing values ("mean", "median", "most_frequent", "constant").
        fill_value: The constant value to fill if strategy is 'constant'.
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    try:
        if strategy == "constant" and fill_value is None:
            return None,1
        
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        return df_imputed,0
    except Exception as e:
        
        return None,2

# Function to handle categorical encoding using Label Encoding
def handle_encoding(df, encoding_type="label", columns=None):
    """
    Handle encoding of categorical variables.
    Args:
        df (pd.DataFrame): The input dataframe.
        encoding_type (str): Encoding type, currently supports "label".
        columns (list): List of categorical columns to encode.
    
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    try:
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if encoding_type == "label":
            for col in columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
        else:
            return None , 1
        
        return df,0
    except Exception as e:
        return None,2

# Function to handle scaling
def handle_scaling(df, scaling_type="standard"):
    """
    Apply scaling to the DataFrame.
    Args:
        df (pd.DataFrame): The input dataframe.
        scaling_type (str): The type of scaling to use, either "standard" or "minmax".
    
    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    try:
        if scaling_type == "standard":
            scaler = StandardScaler()
        elif scaling_type == "minmax":
            scaler = MinMaxScaler()
        else:
            return None,1
        
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled,0
    except Exception as e:
        
        return None,2

# Main function to process data by calling all operations
def preprocess_data(train_data,test_data,missing_strategy, encoding_type, scaling_type, 
                    columns_to_encode=None, fill_value=None):
    """
    Preprocess the data by handling missing values, encoding, and scaling.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        missing_strategy (str): Strategy for handling missing values.
        encoding_type (str): Encoding type for categorical variables.
        scaling_type (str): Scaling type for numerical variables.
        columns_to_encode (list): List of columns to encode. If None, all object columns will be encoded.
        fill_value: The constant value to fill if the missing value strategy is 'constant'.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    try:
        # Step 1: Handle missing values
        df,status = handle_missing_values(df, strategy=missing_strategy, fill_value=fill_value)
        
        # Step 2: Encode categorical columns
        df,status = handle_encoding(df, encoding_type=encoding_type, columns=columns_to_encode)
        
        # Step 3: Scale numerical features
        df,status = handle_scaling(df, scaling_type=scaling_type)
        
        return df,status
    
    except Exception as e:
        return None,1
