"""
Mortality, Migration, and Birth Coefficients by Age Calculation and Visualization
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
# Load data from the database
sys.path.append(r"C:\Users\girli\OneDrive\Desktop\Education_Costs")
from EDA.data_loading import load_data


 # Adjust this path as needed
db_path = r"C:\Users\girli\OneDrive\Desktop\Education_Costs\ds_database.db"
tables= load_data(db_path)  # Call load_data with db_path to establish the query function

def calculate_birth_mortality_migration_birth_rates():
    """Calculate mortality, migration, and birth coefficients by age."""
    # Load data from the database
    birth_df = tables('birth_data_per_region')
    mortality_df = tables('mortality_data_per_region')
    migration_df = tables('migration_data_per_region')
    population_0_16 = tables('population_0_16_per_region')
    population_17_19 = tables('population_17_19_per_region')

    # Merge population data for ages 0-16 and 17-19
    population_df = pd.concat([population_0_16, population_17_19])

    # Calculate total population by region, year, and age
    total_population_by_region_year = population_df.groupby(['Year'])['Total_Population'].sum().reset_index()

    # Merge with birth, mortality, and migration data to compute coefficients
    birth_df = birth_df.merge(total_population_by_region_year, on=['Year'], how='inner')
    birth_df['Birth_Coefficient'] = birth_df['Total_Births'] / birth_df['Total_Population']

    mortality_df = mortality_df.merge(total_population_by_region_year, on=['Year'], how='inner')
    mortality_df['Mortality_Coefficient'] = mortality_df['Total_Deaths'] / mortality_df['Total_Population']

    migration_df = migration_df.merge(total_population_by_region_year, on=['Year'], how='inner')
    migration_df['Migration_Coefficient'] = migration_df['Total_Migrations'] / migration_df['Total_Population']

    # Calculate averages by age
    avg_mortality_rate_by_region = mortality_df.groupby('Region_Code')['Mortality_Coefficient'].mean().reset_index()
    avg_migration_rate_by_region = migration_df.groupby('Region_Code')['Migration_Coefficient'].mean().reset_index()
    avg_birth_coefficients_by_region = birth_df.groupby('Region_Code')['Birth_Coefficient'].mean().reset_index()

    return avg_mortality_rate_by_region, avg_migration_rate_by_region, avg_birth_coefficients_by_region, population_df

def visualize_data(avg_mortality, avg_migration, avg_birth, population_data):
    """Visualize mortality, migration, and birth coefficients by Region."""
    # Mortality, Migration, Birth Rate Visualization
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=avg_mortality['Region_Code'], y=avg_mortality['Mortality_Coefficient'], name="Avg Mortality Coefficient"))
    fig1.add_trace(go.Bar(x=avg_migration['Region_Code'], y=avg_migration['Migration_Coefficient'], name="Avg Migration Coefficient"))
    fig1.add_trace(go.Bar(x=avg_birth['Region_Code'], y=avg_birth['Birth_Coefficient'], name="Avg Birth Coefficient"))
    fig1.update_layout(title="Average Mortality, Migration, and Birth Coefficients by Region", xaxis_title="Region", yaxis_title="Coefficient")
    fig1.show()

    # Total Population of Age Group 0-19 Years Old by Year
    population_by_age = population_data.groupby('Region_Code')['Total_Population'].sum().reset_index()
    fig2 = px.bar(population_by_age, x='Region_Code', y='Total_Population', title="Total Population of Age Group 0-19 Years Old by Year")
    fig2.show()

    # Correlation Plot
    combined_df = pd.merge(avg_birth, avg_mortality, on='Region_Code', how='inner', validate='one_to_one').merge(avg_migration, on='Region_Code', how='inner', validate='one_to_one')
    correlation = combined_df[['Birth_Coefficient', 'Mortality_Coefficient', 'Migration_Coefficient', 'Birth_Coefficient']].corr()
    fig3 = px.imshow(correlation, text_auto=True, title="Correlation Matrix for Mortality, Migration, and Birth Coefficients")
    fig3.show()

    print(combined_df)

    # Display Data Tables Using Plotly
    for title, df in [("Average Mortality Coefficient by Region", avg_mortality),
                      ("Average Migration Coefficient by Region", avg_migration),
                      ("Average Birth Coefficient by Region", avg_birth)]:
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[df[col] for col in df.columns], fill_color='lavender', align='left'))
        ])
        fig.update_layout(title=title)
        fig.show()

        print(df.columns)

def main():
    """Main function for calculating and visualizing mortality, migration, and birth coefficients by age."""
    avg_mortality_rate_by_region, avg_migration_rate_by_region, avg_birth_coefficients_by_region, combined_population = calculate_birth_mortality_migration_birth_rates()
    visualize_data(avg_mortality_rate_by_region, avg_migration_rate_by_region, avg_birth_coefficients_by_region, combined_population)

if __name__ == "__main__":
    main()
