"""Creating a new instance of the SQLite database and inserting data from CSV files."""
import warnings
import sqlite3
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# File paths
birth_data_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\birth_data_per_region.csv"
mortality_data_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\mortality_data_per_region.csv"
migration_data_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\migration_data_per_region.csv"
population_0_16_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\population_0_16_years.csv"
population_17_19_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\population_17_19_years.csv"
primary_costs_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\grundskola_costs_per_child.csv"
secondary_costs_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\database\gymnasieskola_costs_per_child.csv"

# Define regions and their corresponding codes
regions = {
    '01': 'Stockholms län', '03': 'Uppsala län', '04': 'Södermanlands län', '05': 'Östergötlands län',
    '06': 'Jönköpings län', '07': 'Kronobergs län', '08': 'Kalmar län', '09': 'Gotlands län',
    '10': 'Blekinge län', '12': 'Skåne län', '13': 'Hallands län', '14': 'Västra Götalands län',
    '17': 'Värmlands län', '18': 'Örebro län', '19': 'Västmanlands län', '20': 'Dalarnas län',
    '21': 'Gävleborgs län', '22': 'Västernorrlands län', '23': 'Jämtlands län', '24': 'Västerbottens län',
    '25': 'Norrbottens län'
}

# Create a connection to the SQLite database
conn = sqlite3.connect("ds_database.db")
cur = conn.cursor()

def create_tables():
    """
    Create the necessary tables in the SQLite database if they don't already exist.
    """
    cur.execute("DROP TABLE IF EXISTS birth_data_per_region")
    cur.execute('''CREATE TABLE birth_data_per_region(
                    Region TEXT,
                    Year INTEGER,
                    Total_Births INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS mortality_data_per_region")
    cur.execute('''CREATE TABLE mortality_data_per_region(
                    Region TEXT,
                    Age INTEGER,
                    Year INTEGER,
                    Total_Deaths INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS migration_data_per_region")
    cur.execute('''CREATE TABLE migration_data_per_region(
                    Region_Code TEXT,
                    Region_Name TEXT,
                    Age INTEGER,
                    Year INTEGER,
                    Total_Migrations INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS population_0_16_per_region")
    cur.execute('''CREATE TABLE population_0_16_per_region(
                    Region TEXT,
                    Age INTEGER,
                    Year INTEGER,
                    Total_Population INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS population_17_19_per_region")
    cur.execute('''CREATE TABLE population_17_19_per_region(
                    Region TEXT,
                    Age INTEGER,
                    Year INTEGER,
                    Total_Population INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS grundskola_costs_per_region")
    cur.execute('''CREATE TABLE grundskola_costs_per_region(
                    Year INTEGER,
                    Fixed_Cost_Per_Child_SEK INTEGER,
                    Current_Cost_Per_Child_SEK INTEGER)''')

    cur.execute("DROP TABLE IF EXISTS gymnasieskola_costs_per_region")
    cur.execute('''CREATE TABLE gymnasieskola_costs_per_region(
                    Year INTEGER,
                    Fixed_Cost_Per_Child_SEK INTEGER,
                    Current_Cost_Per_Child_SEK INTEGER)''')

def map_region_code(df, code_column='Region'):
    """
    Map region codes to region names in a DataFrame, adding Region column.
    """
    if code_column in df.columns:
        df['Region'] = df[code_column].map(regions)
    else:
        raise KeyError(f"'{code_column}' column not found in DataFrame.")
    return df

def load_and_insert_data():
    """
    Load data from CSV files, map region codes to names, and insert into the database tables.
    """
    # Load CSV files into pandas dataframes and map region names
    birth_data = pd.read_csv(birth_data_path, encoding='ISO-8859-1')
    mortality_data = pd.read_csv(mortality_data_path, encoding='ISO-8859-1')

    # Load migration data with semicolon delimiter
    migration_data = pd.read_csv(migration_data_path, sep=';', encoding='ISO-8859-1', skip_blank_lines=True)

    # Print the columns to check for length mismatch
    print("Migration data columns before renaming:", migration_data.columns.tolist())

    # Clean up migration_data by dropping any rows that might be empty or have NaN
    migration_data.dropna(how='all', inplace=True)  # Drop fully empty rows

    # Rename columns if necessary to match the database schema
    # Ensure the number of new names matches the number of columns in migration_data
    if migration_data.shape[1] == 6:  # Check if there are 6 columns
        migration_data.columns = ['Region_Code', 'Region_Name', 'Age', 'Year', 'Total_Migrations', 'Unused_Column']
        migration_data = migration_data.drop(columns=['Unused_Column'])  # Drop the last column if not needed
    elif migration_data.shape[1] == 5:  # If there are 5 columns, use the original names
        migration_data.columns = ['Region_Code', 'Region_Name', 'Age', 'Year', 'Total_Migrations']
    else:
        raise ValueError(f"Unexpected number of columns: {migration_data.shape[1]}")

    # Load other dataframes
    population_0_16 = pd.read_csv(population_0_16_path, encoding='ISO-8859-1')
    population_17_19 = pd.read_csv(population_17_19_path, encoding='ISO-8859-1')
    primary_costs = pd.read_csv(primary_costs_path, encoding='ISO-8859-1')
    secondary_costs = pd.read_csv(secondary_costs_path, encoding='ISO-8859-1')

    # Insert data into tables
    birth_data.to_sql('birth_data_per_region', conn, if_exists='append', index=False)
    mortality_data.to_sql('mortality_data_per_region', conn, if_exists='append', index=False)
    migration_data.to_sql('migration_data_per_region', conn, if_exists='append', index=False)
    population_0_16.to_sql('population_0_16_per_region', conn, if_exists='append', index=False)
    population_17_19.to_sql('population_17_19_per_region', conn, if_exists='append', index=False)
    primary_costs.to_sql('grundskola_costs_per_region', conn, if_exists='append', index=False)
    secondary_costs.to_sql('gymnasieskola_costs_per_region', conn, if_exists='append', index=False)

# Create tables in the SQLite database
create_tables()

# Insert data from CSV files into the database
load_and_insert_data()

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database created and data inserted successfully.")
