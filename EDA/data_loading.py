"""Load data from a SQLite database."""
import sqlite3
import pandas as pd

def load_data(db_path):
    """Load data from a SQLite database"""
    def query(tables):
        """Query the database for the specified tables"""
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {tables}", conn)
        conn.close()
        return df
    return query
