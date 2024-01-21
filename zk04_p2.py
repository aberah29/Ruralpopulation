import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter

df_countries=pd.read_csv('API_SP.RUR.TOTL_DS2_en_csv.csv',skiprows=4)

def plot_rural_population_prediction(country_name, indicator_name, df_countries):
    # Extract years and Rural population data for the specified country and indicator
    country_data = df_countries[(df_countries['Country Name'] 
                                 ==country_name) & (df_countries['Indicator Name']
                                                    ==indicator_name)]
    years = country_data.columns[4:]  # Assuming the years start from the 5th column
    rural_pop = country_data.iloc[:, 4:].values.flatten()

    # Convert years to numeric values
    years_numeric = pd.to_numeric(years, errors='coerce')
    rural_pop = pd.to_numeric(rural_pop, errors='coerce')

    # Remove rows with NaN or inf values
    valid_data_mask = np.isfinite(years_numeric) & np.isfinite(rural_pop)
    years_numeric = years_numeric[valid_data_mask]
    rural_pop = rural_pop[valid_data_mask]

    # Define the model function
    def rural_pop_model(year, a, b, c):
        return a * np.exp(b * (year - 1990)) + c

    # Curve fitting with increased maxfev
    params, covariance = curve_fit(rural_pop_model, years_numeric, rural_pop, 
                                   p0=[1, -0.1, 90], maxfev=10000)

    # Optimal parameters
    a_opt, b_opt, c_opt = params

    # Generate model predictions for the year 2040
    year_2040 = 2040
    rural_pop_2040 = rural_pop_model(year_2040, a_opt, b_opt, c_opt)

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(years_numeric, rural_pop, label
                ='Original Data', color='skyblue', alpha=0.7, 
                edgecolors='navy', linewidths=0.7)
    plt.plot(years_numeric, rural_pop_model(years_numeric, a_opt, b_opt, c_opt), 
             label='Fitted Curve', color='salmon', linewidth=2)

    # Highlight the prediction for 2040
    plt.scatter(year_2040, rural_pop_2040, color='limegreen', marker='*',
                label='Prediction for 2040', s=100, edgecolors='black')

    # Add labels and legend
    plt.title(f'Rural Population Prediction for {country_name}', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Rural Population', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Beautify the plot
    plt.style.use('ggplot')
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage:
countries = ['Bahrain', 'India', 'Belgium']
indicator_name = 'Rural population'

for country in countries:
    plot_rural_population_prediction(country, indicator_name, df_countries)

# Extract data for the years 1970 and 2020
years = ['1970', '2019']
rural_pop = df_countries[['Country Name'] + years]

# Drop rows with missing values
rural_pop = rural_pop.dropna()

# Set 'Country Name' as the index
rural_pop.set_index('Country Name', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(rural_pop)

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
rural_pop['Cluster'] = labels

# Define a custom formatter to display numbers in thousands
def format_thousands(x, pos):
    return f'{int(x / 1000)}K'

# Custom color map for clusters
colors = np.array(['red', 'blue'])

# Visualize the clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Cluster for 1970
axs[0].scatter(rural_pop[years[0]], rural_pop.index, c=colors[labels], cmap='viridis')
axs[0].set_title(f'Rural Population in {years[0]}')
axs[0].set_xlabel('Rural Population')
axs[0].set_ylabel('Countries')
axs[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom formatter to x-axis

# Cluster for 2020
axs[1].scatter(rural_pop[years[1]], rural_pop.index, c=colors[labels], cmap='viridis')
axs[1].set_title(f'Rural Population in {years[1]}')
axs[1].set_xlabel('Rural Population')
axs[1].set_ylabel('Countries')
axs[1].xaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom formatter to x-axis

# Manually set y-axis label
for ax in axs:
    ax.set_yticks([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.show()


