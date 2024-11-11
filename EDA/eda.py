import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data(query):
    database_path = r'C:\Users\girli\OneDrive\Desktop\Education_Costs\ds_database.db'
    conn = None
    try:
        conn = sqlite3.connect(database_path)
        data = pd.read_sql_query(query, conn)
        return data
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    finally:
        if conn is not None:
            conn.close()

def perform_eda(query):
    # Fetch the data
    data = fetch_data(query)

    if data.empty:
        print("No data fetched. EDA cannot be performed.")
        return

    # Display the first few rows of the DataFrame
    print("First few rows of the data:")
    print(data.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(data.describe())

    # Checking for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())

    # Distribution of numerical features
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # Box plots to identify outliers
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

    # Correlation heatmap
    correlation_matrix = data[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()

# Example query to fetch data
query = "SELECT * FROM migration_data_per_region;"
perform_eda(query)
