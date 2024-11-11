import pandas as pd
import sqlite3

def add_age_column_to_birth_data():
    conn = sqlite3.connect('ds_database.db')

    # Load birth data
    birth_data = pd.read_sql_query("SELECT Region_Code, Region_Name, Year, Total_Births FROM birth_data_per_region", conn)

    # Load population data for ages 0-16 and 17-19
    population_0_16 = pd.read_sql_query("SELECT Region_Code, Region_Name, Year, Age, Total_Population FROM population_0_16_per_region", conn)
    population_17_19 = pd.read_sql_query("SELECT Region_Code, Region_Name, Year, Age, Total_Population FROM population_17_19_per_region", conn)

    # Concatenate both population tables to have all ages in one DataFrame
    population_data = pd.concat([population_0_16, population_17_19])

    # Merge birth data with population data on Region_Code, Region_Name, and Year
    merged_data = pd.merge(birth_data, population_data, on=["Region_Code", "Region_Name", "Year"], how="inner")

    # Calculate total population per region per year for ages 0-19 to determine the proportion
    total_population = merged_data.groupby(['Region_Code', 'Region_Name', 'Year'])['Total_Population'].transform('sum')

    # Calculate the birth count for each age by proportionally distributing based on population count
    merged_data['Age_Birth_Count'] = (merged_data['Total_Population'] / total_population) * merged_data['Total_Births']

    # Drop unnecessary columns and rename as needed
    age_specific_birth_data = merged_data[['Region_Code', 'Region_Name', 'Year', 'Age', 'Age_Birth_Count']].copy()
    age_specific_birth_data.rename(columns={'Age_Birth_Count': 'Total_Births'}, inplace=True)

    conn.close()
    return age_specific_birth_data

# Run the function and print the result to verify
age_specific_birth_data = add_age_column_to_birth_data()
print(age_specific_birth_data.head())

def save_age_specific_birth_data(dataframe, db_path='ds_database.db'):
    conn = sqlite3.connect(db_path)
    dataframe.to_sql('birth_data_per_region', conn, if_exists='replace', index=False)
    conn.close()
    print("Data saved to 'birth_data_per_region' table in the database.")

# Call the save function
save_age_specific_birth_data(age_specific_birth_data)
