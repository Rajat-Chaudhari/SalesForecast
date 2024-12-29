import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

import numpy as np
import pandas as pd
import csv
import joblib
import datetime
import mlflow
import os
import sys
import traceback
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from result_saver import save_results, save_performance_metrics, save_model
from hyperparameter_tunning import generate_param_combinations

# Model definition
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation_function):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.activation = activation_function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Helper functions
def get_activation_function(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
def mape_loss(predictions, targets):
    epsilon = 1e-7  # Small constant to avoid division by zero
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) 

def get_loss_function(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "mape":
        # Return a lambda that calls mape_loss to make it compatible with PyTorch's loss functions
        return lambda predictions, targets: mape_loss(predictions, targets)
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_optimizer(name, model_parameters, lr):
    if name == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif name == "sgd":
        return optim.SGD(model_parameters, lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def prepare_dataloader(data, target_col, batch_size,validation_ratio):
    print("Data Preparation Started .........")
    # Calculate indices for the splits
    train_ratio= 1 - (validation_ratio*2)
    dataset_size = len(data)
    # train_size = int(dataset_size * train_ratio)
    # val_size = int(dataset_size * validation_ratio)
    # test_size = dataset_size - train_size - val_size
    
    train_size = 64
    val_size = 4
    test_size = 4
    
    targets = data[[target_col]]
    data = data.drop(columns=[target_col])
    
    n_size = data.shape[1]
    
    print(f'data :{data.columns} || shape : {data.shape} || target : {targets.columns} || shape : {targets.shape}')
    
    # Split data and targets sequentially
    train_data, train_targets = data[:train_size], targets[:train_size]
    val_data, val_targets = data[train_size:train_size + val_size], targets[train_size:train_size + val_size]
    test_data, test_targets = data[train_size + val_size:], targets[train_size + val_size:]

    print(f'Train Data : {train_data.shape} || Train Target : {train_targets.shape}')
    print(f'Test Data : {test_data.shape} || Test Target : {test_targets.shape}')
    print(f'Valid Data : {val_data.shape} || Valid Target : {val_targets.shape}')
    

    train_dataset = TensorDataset(torch.tensor(train_data.values, dtype=torch.float32), torch.tensor(train_targets.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data.values, dtype=torch.float32), torch.tensor(val_targets.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_data.values, dtype=torch.float32), torch.tensor(test_targets.values, dtype=torch.float32))

    # DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader,n_size


# Function to log losses to a CSV file
def log_losses_to_csv(epoch, train_loss, val_loss,run_id):
    
    file_path=f'training_losses_{run_id}.csv'
    
    # If it's the first epoch, write the header
    if epoch == 1:
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
    
    # Append the losses for the current epoch
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([epoch, train_loss, val_loss])
        
    mlflow.log_artifact(file_path)

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, save_path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if current validation loss is an improvement
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(model)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print("Validation loss decreased, saving model")

# Training function
def train_model(model, train_loader, val_loader, 
                loss_function,optimizer,early_stopping,epoch,run_id):
    # Training loop
    for epoch in range(1, epoch + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        # Log the training and validation loss in CSV file
        log_losses_to_csv(epoch, train_loss, val_loss,run_id)

        print(f'Epoch [{epoch}/{epoch}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if eval(str(early_stopping.patience)) is not None:
            # Check for early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
    return model

def build_fully_Connected_nn(train_loader, val_loader,target_col, params,n_size,run_id):
    # Instantiate the model with dynamic activation function
    activation_function = get_activation_function(params['activation_function'])
    model = FullyConnectedNN(n_size, params['hidden_size1'], params['hidden_size2'], params['output_size'], activation_function)
    loss_function = get_loss_function(params['loss_function'])
    optimizer = get_optimizer(params['optimizer'], model.parameters(), params['learning_rate'])
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=params['early_stopping_patience'], verbose=True)
    epoch = params['num_epochs']
    
    
    # Train the model
    fitted_model = train_model(model, train_loader, val_loader, 
                               loss_function,optimizer,early_stopping,epoch,run_id)
    
    return  fitted_model 

def get_predictions_and_targets(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actual_targets = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, targets in data_loader:  # Retrieve both inputs and targets
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())       # Store predictions
            actual_targets.extend(targets.cpu().numpy())    # Store actual targets

    return predictions, actual_targets  # Lists of predictions and targets

def run_fully_connected_nn(model_name,train_data, test_data, target_col, params):
    """Run Fully Connected NN model."""
    
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
                
                run_id  = run.info.run_id
                
                print(f'model name : {model_name} train data size : {train_data.shape} || target_col : {target_col} || params : {params} || run id : {run_id}')

                data = pd.concat([train_data, test_data],axis=0)
                
                
                
                # Prepare dataloaders
                train_loader, val_loader, test_loader,n_size = prepare_dataloader(data, target_col, params['batch_size'], params['validation_split'])
                
                fitted_model = build_fully_Connected_nn(train_loader, val_loader,target_col, params,n_size,run_id)
                
                prediction_train_lst,actual_train_lst = get_predictions_and_targets(fitted_model, train_loader)
                prediction_test_lst,actual_test_lst = get_predictions_and_targets(fitted_model,test_loader)
                
                prediction_val_lst,actual_val_lst = get_predictions_and_targets(fitted_model,val_loader)
                
                print("Val Actual",actual_val_lst)
                print("val Pred",prediction_val_lst)
                
                save_performance_metrics(actual_train_lst, actual_test_lst,
                                        prediction_train_lst, prediction_test_lst,
                                        model_name,params,run_id)
                
                
                
                save_model(fitted_model,model_name,run_id)
                
    except Exception as e:
        print("Error ::",traceback.format_exc())
        raise e
   
  
  
########## Testing Purpose #############
   
# model_name = 'FullyConnectedNN'
# params = {
#     'input_size': [4],                     
#     'hidden_size1': [64],                   
#     'hidden_size2': [32],                   
#     'output_size': [1],                     
#     'batch_size': [32],                     
#     'num_epochs': [10],                    
#     'early_stopping_patience': [10],        
#     'loss_function': ['mse'],               
#     'optimizer': ['adam'],                  
#     'learning_rate': [0.001],               
#     'activation_function': ['relu'],        
#     'validation_split': [0.2]               
# }


# data = pd.read_csv('../AOD_Test_Dataset.csv')
# data['Date'] = data['Date'].str.strip()
# data['Date'] = pd.to_datetime(data['Date'], format='%b-%y') 
# data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
# train_size = 80
# target_col = 'Sales'

# train_data, test_data = data[:train_size], data[train_size:]

# train_data['Month'] = pd.to_datetime(train_data['Date']).dt.month
# train_data.drop(columns=['Date'], inplace=True)

# test_data['Month'] = pd.to_datetime(test_data['Date']).dt.month
# test_data.drop(columns=['Date'], inplace=True)

# print("Model Started ..........")
# print(f'Data : {data.shape} || Train Data :{train_data.shape} || Test Data : {test_data.shape}')
# print(f'Data Columns : {data.columns}')

# run_fully_connected_nn(model_name,train_data, test_data, target_col, params)


   