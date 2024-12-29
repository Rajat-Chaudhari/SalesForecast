from itertools import product
import traceback

def generate_param_combinations(param_grid):
    """
    Generate all combinations of hyperparameters from a given parameter grid.
    
    Parameters:
    - param_grid (dict): Dictionary where keys are parameter names and values are either
                         a list specifying [start, end] for ranges or a list of discrete values.
    
    Returns:
    - list of dict: List of dictionaries, each containing one combination of parameters.
    """
    print("Started parameter combination creation...")
    
    # Adjust each parameter's range or use discrete values directly
    adjusted_param_grid = {}
    for key, value in param_grid.items():
        # Only treat as a range if both values are integers and not Booleans
        if (
            isinstance(value, list) 
            and len(value) == 2 
            and all(isinstance(i, int) and not isinstance(i, bool) for i in value)
        ):
            adjusted_param_grid[key] = list(range(value[0], value[1]))
        else:
            adjusted_param_grid[key] = value  # Keep discrete values directly, like [True, False]

    # Debugging output to verify the adjusted parameter grid
    print("Adjusted Parameter Grid:", adjusted_param_grid)

    # Extract parameter names and values
    param_names = list(adjusted_param_grid.keys())
    param_values = list(adjusted_param_grid.values())
    
    # Generate all possible combinations
    combinations = list(product(*param_values))
    print("Generated combinations:", combinations)

    # Convert each combination into a dictionary
    combination_dicts = [dict(zip(param_names, combo)) for combo in combinations]

    return combination_dicts
    

        
