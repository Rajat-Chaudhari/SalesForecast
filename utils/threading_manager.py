import mlflow
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_models_in_batches(model_functions, batch_size):
    """
    Run models in batches using threading.
    
    Args:
        model_functions (list): List of model functions to execute.
        batch_size (int): Number of models to run concurrently.
    """
    total_models = len(model_functions)
    print(f"Total models to run: {total_models}")
    
    for i in range(0, total_models, batch_size):
        batch = model_functions[i:i + batch_size]
        print(f"Running batch {i // batch_size + 1}")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(model) for model in batch]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Model completed: {result}")
                except Exception as e:
                    print(f"Model failed with error: {e}")
        
        print(f"Batch {i // batch_size + 1} completed.")
        
        
def run_models_sequentially(model_functions):
    """
    Run models sequentially (one by one) without threading.
    
    Args:
        model_functions (list): List of model functions to execute.
    """
    try :
        
        total_models = len(model_functions)
        print(f"Total models to run: {total_models}")
        for idx, model in enumerate(model_functions, 1):
            print(f"Running model {idx}/{total_models}...")
            result = model()  # Execute the model function
            print(f"Model {idx} completed successfully: {result}")
            
        return 0
          
    except Exception as e:
        return 6
        
        

