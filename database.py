import traceback
import psycopg2
from psycopg2 import sql
from config import load_config
from datetime import datetime

def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # Connecting to the PostgreSQL server
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Connection error: {error}")
        return None  # Return None if connection fails

def insert_details(connection, table_name, columns, values):
    """
    Insert details into a specified table in the PostgreSQL database.
    
    Parameters:
        connection (psycopg2 connection): A connection to the PostgreSQL database.
        table_name (str): The name of the table to insert into.
        columns (list of str): The column names for the data to be inserted.
        values (list): The values to insert into each column.

    Returns:
        str: "Successfully inserted" if the insertion is successful, otherwise "Insertion failed".
    """
    
    try:
        # Ensure that we have equal columns and values
        if len(columns) != len(values):
            return "Column and value counts do not match."

        # Prepare the SQL query dynamically
        query = sql.SQL("INSERT INTO {table} ({fields}) VALUES ({placeholders});").format(
            table=sql.Identifier(table_name),
            fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(', ').join(sql.Placeholder() * len(values))
        )
        
        print("Query to execute:", query.as_string(connection))  # Print the query for debugging
        
        with connection.cursor() as cursor:
            cursor.execute(query, values)  # Pass values as a list
            connection.commit()  # Commit the transaction
            return "Successfully inserted"
        
    except (Exception, psycopg2.DatabaseError) as error:
        connection.rollback()  # Rollback in case of error
        print("Error occurred:", traceback.format_exc())
        return "Insertion failed"

# if __name__ == '__main__':
#     config = load_config()  # Load database configuration
#     print("Database configuration loaded:", config)

#     # Connect to the database
#     connection = connect(config)
    
#     if connection:  # Proceed only if connection was successful
#         table_name = 'insight_301024_tbl'
#         columns = ['session_id', 'file_id', 'status', 'insertion_time']
        
#         # Get current date and time without timezone info
#         current_datetime = datetime.now()
#         values = ['abc', 'xyz', 'enable', current_datetime]
        
#         # Insert details into the table
#         status = insert_details(connection, table_name, columns, values)
#         print(status)
        
#         connection.close()  # Close the connection after the operation
#     else:
#         print("Failed to connect to the database.")
