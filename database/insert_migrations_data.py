"""Insert Migrations data into the database"""

import pandas as pd
import sqlite3

# Path to the CSV file
csv_file_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\migration_data_per_region.csv"

# Connect to the SQLite database
conn = sqlite3.connect('ds_database.db')  # Replace with your actual database path if different

# Load CSV data with pandas, assuming semicolon delimiter
data = pd.read_csv(csv_file_path, delimiter=';')

# Insert data into the existing 'migration_data_per_region' table in SQLite
data.to_sql('migration_data_per_region', conn, if_exists='append', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()
