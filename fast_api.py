from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import time
from main import start_model_training,get_model_details
from graphs import process_file
from fastapi.responses import JSONResponse
from typing import List


app = FastAPI()

# Directory to save the CSV file
DATA_DIR = "train_data"
os.makedirs(DATA_DIR, exist_ok=True)




# Pydantic model to define expected JSON input structure
class UserInput(BaseModel):
    file_path:str
    target_col: str
    date_col: str
    model_type: str
    mode: str
    training_data_size: int
    freq: str
    selected_model_lst: List[str]
    model_kpi:str
    kpi_range:str
    session_id:str

# 1. API For Model training
# POST API to accept CSV and other input parameters
@app.post("/start_model_training/")
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...,description="Upload csv file only"),  # CSV File upload
    target_col: str = Form(...,description="Give name of your target column"),
    date_col: str = Form(...,description="Give date column name"),
    model_type: str = Form(...,description="Give Model type either univariate or multivariate"),
    mode: str = Form(...,description="Give model mode either auto or manual"),
    training_data_size: int = Form(...,description="Give training data size in number of month"),
    freq: str = Form(...,description="Specify the data frequency like M for month , W for week etc"),
    selected_model_lst: List[str] = Form(...,description="Provide algorithm name"),
    model_kpi:str = Form(...,description="Provide name from mentioned text : MAE , MAPE , MSE"),
    kpi_range:str = Form(...,description="Provide range in 0 - 1 and give format 0.05 - 0.07"),
    session_id:str = Form(...,description="Provide session id")
):
    # Save the uploaded file locally
    file_path = f"{DATA_DIR}/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Collect the rest of the parameters into a Pydantic model
    user_input = UserInput(
        file_path = file_path,
        target_col=target_col,
        date_col=date_col,
        model_type=model_type,
        mode=mode,
        training_data_size=training_data_size,
        freq=freq,
        selected_model_lst=selected_model_lst,
        model_kpi=model_kpi,
        kpi_range=kpi_range,
        session_id = session_id
    )

    model_dict = {
    "file_path" : file_path,
    "target_col" : target_col,
    "date_col" : date_col,
    "model_type" : model_type,
    "mode" : mode,
    "training_data_size" : training_data_size,
    "freq" : freq ,
    "selected_model_lst" : [model.strip() for model in selected_model_lst[0].split(',')],
    "model_kpi":model_kpi,
    "kpi_range":kpi_range,
    "session_id":session_id
}
    # Start background task for training the model
    background_tasks.add_task(start_model_training, model_dict)

    return {"message": "Model training started in the background", "file_path": file.filename}


# Define a model for input parameters
class ModelSummaryRequest(BaseModel):
    experiment_name: str
    run_name: str
    session_id : str

@app.post("/get_model_summary/")
async def get_model_summary(experiment_name: str = Form(...,description="Provide valid experiment name"),
    run_name: str = Form(...,description="Provide valid run name"),
    session_id: str = Form(...,description="Provide valid session id")):

    model_summary = ModelSummaryRequest(experiment_name = experiment_name,
                                              run_name=run_name,session_id=session_id)
    
    response = get_model_details(experiment_name,run_name,session_id)
    
    return response


@app.post("/get_insights/")
async def get_insight(
    session_id: str = Form(..., description="Session ID for the upload"),
    files: List[UploadFile] = File(..., description="Upload one or multiple CSV/XLSX files")
):
    overall_response = {}

    for idx, file in enumerate(files, start=1):
        
        # Process each file individually
        file_response = process_file(file.file, file.filename, session_id)
        overall_response[f"{idx}"] = file_response
        
    
    return JSONResponse(content=overall_response)



# To run the FastAPI app, use the command:
# uvicorn your_filename:app --reload


