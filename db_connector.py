import mysql.connector
from mysql.connector import Error

def insert_into_table():
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host='127.0.0.1:3307',   
            database='chat',  
            user='root', 
            password='password'  
        )

        if connection.is_connected():
            print("Successfully connected to the database")
            
            # Create a cursor object
            cursor = connection.cursor()

            # SQL query to insert data
            insert_query = """INSERT INTO your_table_name (column1, column2, column3)
                              VALUES (%s, %s, %s)"""
            
            # Data to insert
            record = ('value1', 'value2', 'value3')
            
            # Execute the insert query
            cursor.execute(insert_query, record)
            
            # Commit the transaction to save changes
            connection.commit()
            
            print(f"{cursor.rowcount} record inserted successfully.")

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


