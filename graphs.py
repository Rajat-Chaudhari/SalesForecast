import pandas as pd
import json
from fastapi import HTTPException
from itertools import combinations
from ollama import chat  # Assuming ollama.chat is available
import re
import traceback
import psycopg2
from psycopg2 import sql
from config import load_config
from datetime import datetime
from database import connect,insert_details

# Load exception messages from exception.json
with open("exception.json") as f:
    exceptions = json.load(f)

def read_file(file, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX file into a DataFrame and validate its type."""
    try:
        if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
            return None,2
        
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            
        return df,0
    
    except Exception as e:
        print(traceback.format_exc())
        return None,6
        
def validate_and_clean_df(df: pd.DataFrame):
    """Validate DataFrame and exclude columns with unique values."""
    try:
        non_unique_columns = [col for col in df.columns if df[col].nunique() < len(df)]
        
        if len(non_unique_columns) < 2:
            return None,9
        
        return df[non_unique_columns],0
    
    except Exception as e:
        print(traceback.format_exc())
        return None,6

def generate_graph_data(df: pd.DataFrame, x_col: str, y_col: str):
    """Generate graph data to return as JSON format."""
    try:
        
        x_data = df[x_col].tolist()
        y_data = df[y_col].tolist()
        return {
            "x_data": x_data,
            "y_data": y_data,
            "x_axis_label": x_col,
            "y_axis_label": y_col
        },0
        
    except Exception as e:
        print(traceback.format_exc())
        return None,6



def generate_insights(graph_json: dict, user: str, industry: str, client: str) -> dict:
    """Generate insights using ollama model based on graph JSON data."""
    try:
        
        # Generate insights using the chat model
        response = chat(
            model='llama3.1',
            messages=[{
                'role': 'user',
                'content': f"""
                    Act as a highly skilled {user} specializing in the {industry} industry, providing data-driven insights for {client} based on the graph data provided.

                    Guidelines:
                    1. Analyze each chart provided and generate insightful findings specifically applicable to {industry} and {client}.
                    2. Each insight should include relevant details in JSON format with keys "Insight", "Rationale", and "Implication".
                    3. Only infer insights directly from the graph data.
                    4. Present the response in a structured JSON format for easy consumption.
                    5. Ensure the response is in simple plain text without any markdown, special characters (e.g., *, **, \n), or bullet points. Each insight should be formatted as a clear sentence within the JSON fields, without extra formatting or unnecessary line breaks.

                    Graph Data: {json.dumps(graph_json)}"""  # Serialize graph_json to a JSON string
            }]
        )
        
        # Extract the content from the response
        content = response['message']['content']

        # Use regex to clean the response
        cleaned_text = re.sub(r'[\*\n]+', ' ', content)  # Remove special characters and new lines
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize spaces and strip leading/trailing spaces


        return cleaned_text,0  # Return the structured dictionary
    
    except Exception as e:
        print(traceback.format_exc())
        return None,6

def process_file(file, filename, session_id):
    """Process a single file to generate graph data and insights for each column combination."""
    try:
        
        print(f"File Name : {filename}")
        
        # Read and validate the file
        df,status = read_file(file, filename)
        
        if status != 0:
            return exceptions['file_not_found'][str(status)]
        
        print("First Read")
        print(df.columns)
        print(df.shape)
        
        df,status = validate_and_clean_df(df)
        if status != 0:
            return exceptions['file_not_found'][str(status)]
        
        print(df.columns)
        print(df.shape)
        
        user, industry, client = 'Data Analysts','Automobile','Toyota'
        
        # Prepare the response dictionary for each file
        file_response = {}

        # Generate all column combinations for graphs
        combination_counter = 1
        graph_json = {}  # To accumulate data for insights generation

        for x_col, y_col in combinations(df.columns, 2):
            
            print("x_col :",x_col,"y_col :",y_col)
            # Determine chart type based on y_col data type
            
            chart_type = "line" if pd.api.types.is_numeric_dtype(df[y_col]) else "bar"

            # Generate graph data
            graph_data,status = generate_graph_data(df, x_col, y_col)
            
            if status != 0:
                return exceptions['file_not_found'][str(status)]
            
            graph_json[f"{combination_counter}"] = {
                "data": [graph_data],
                "graph_name": f"{chart_type.capitalize()} Chart of {y_col} over {x_col}",
                "graph_type": chart_type
            }
            combination_counter += 1

        # Generate insights based on entire graph JSON
        insights,status = generate_insights(graph_json, user, industry, client)
        if status != 0:
            return exceptions['file_not_found'][str(status)]

        # Append insights to the response for each chart
        for key, value in graph_json.items():
            file_response[key] = {
                "data": value["data"],
                "insights": insights,
                "graph_name": value["graph_name"],
                "graph_type": value["graph_type"]
            }
            
            ####
            config = load_config()  # Load database configuration
            print("Database configuration loaded:", config)

            # Connect to the database
            connection = connect(config)
            
            if connection:  # Proceed only if connection was successful
                table_name = 'insight_301024_tbl'
                
                columns = ['session_id', 'file_id', 'status', 'insertion_time']
        
                # Get current date and time without timezone info
                current_datetime = datetime.now()
                values = [session_id, filename, 'enable', current_datetime]
                
                # Insert details into the table
                status = insert_details(connection, table_name, columns, values)
                print(status)
                
                table_name = 'insight_summary_301024_tbl'
                # Insert details into the table
                columns = ['chart_name','summary_data','session_id','file_id','status','insertion_time']
                
                
              
                values = [value["graph_type"],
                        json.dumps({"data": value["data"],"insights": insights,"graph_name": value["graph_name"]}),
                          session_id,
                          filename,
                          'enable',
                          datetime.now()]
                
                status = insert_details(connection, table_name, columns, values)
                
                connection.close()  # Close the connection after the operation
            else:
                print("Failed to connect to the database.")


        return file_response
    
    except Exception as e:
        print(traceback.format_exc())
        return exceptions['file_not_found'][str(6)]
        
