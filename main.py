import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

### PART 1 : Dataset Overview ###

# Load the dataset
df_area = pd.read_csv('area.csv')
df_delta_storage = pd.read_csv('delta_storage.csv')
df_storage = pd.read_csv('storage.csv')
df_evaporation = pd.read_csv('evaporation.csv')
df_inflow = pd.read_csv('inflow.csv')
df_inflow_volume = pd.read_csv('inflow_volume.csv')
df_total_release = pd.read_csv('total_release.csv')
df_release_volume = pd.read_csv('release_volume.csv')
df_unregulated_inflow_volume = pd.read_csv('unregulated_inflow_volume.csv')
df_unregulated_inflow = pd.read_csv('unregulated_inflow.csv')
df_mod_unregulated_inflow = pd.read_csv('mod_unregulated_inflow.csv')
df_mod_unregulated_inflow_volume = pd.read_csv('mod_unregulated_inflow_volume.csv')

# listing data frames
dataframes = {
    'Area': df_area,
    'Delta Storage': df_delta_storage,
    'Storage': df_storage,
    'Evaporation': df_evaporation,
    'Inflow': df_inflow,
    'Inflow Volume': df_inflow_volume,
    'Total Release': df_total_release,
    'Release Volume': df_release_volume,
    'Unregulated Inflow Volume': df_unregulated_inflow_volume,
    'Unregulated Inflow': df_unregulated_inflow,
    'Modified Unregulated Inflow': df_mod_unregulated_inflow,
    'Modified Unregulated Inflow Volume': df_mod_unregulated_inflow_volume
}

### PART 2 : Data Frame Visualization ###
# Plot histograms for all numerical columns in each dataframe
for name, df in dataframes.items():
    # Create histogram for each dataframe
    df.hist(bins=20, figsize=(10, 8))
    plt.suptitle(f'Histograms for {name} Dataframe', fontsize=16)
    plt.show()

# Pairplot for exploring relationships between numerical variables
for name, df in dataframes.items():
    sns.pairplot(df)
    plt.suptitle(f'Pairplot for {name} Dataframe', y=1.02)
    plt.show()
    
### PART 3 : Merging Data Sets ###

# Convert 'datetime' column to datetime format
dfs = [df_area, df_storage, df_delta_storage, df_inflow, df_inflow_volume, 
       df_unregulated_inflow, df_unregulated_inflow_volume, df_mod_unregulated_inflow, 
       df_mod_unregulated_inflow_volume, df_evaporation, df_release_volume, df_total_release]

for df in dfs:
    df['datetime'] = pd.to_datetime(df['datetime'])

# Merge datasets on 'datetime' keeping all data (outer join)
df_merged = df_area.merge(df_storage, on='datetime', how='outer') \
                   .merge(df_delta_storage, on='datetime', how='outer') \
                   .merge(df_inflow, on='datetime', how='outer') \
                   .merge(df_inflow_volume, on='datetime', how='outer') \
                   .merge(df_unregulated_inflow, on='datetime', how='outer') \
                   .merge(df_unregulated_inflow_volume, on='datetime', how='outer') \
                   .merge(df_mod_unregulated_inflow, on='datetime', how='outer') \
                   .merge(df_mod_unregulated_inflow_volume, on='datetime', how='outer') \
                   .merge(df_evaporation, on='datetime', how='outer') \
                   .merge(df_release_volume, on='datetime', how='outer') \
                   .merge(df_total_release, on='datetime', how='outer')

# Move 'datetime' to the first column
df_merged = df_merged[['datetime'] + [col for col in df_merged.columns if col != 'datetime']]

# Save the merged DataFrame to a new CSV file
df_merged.to_csv('merged_dataset.csv', index=False)

# Display the first few rows of the merged DataFrame
print(df_merged.sample(n=10))

### PART 4 : Handling Missing Data ###

# Heatmap to visualize missing data
sns.heatmap(df_merged.isnull(), cbar=False, cmap='viridis')
plt.show()

# Filtering the merged data set to keep the cleaned data 

# Remove rows from '2006-10-02' onwards
df_cleaned = df_merged[df_merged['datetime'] > '2006-10-02']

# Display the filtered DataFrame
print(df_cleaned.head())

# Save the filtered DataFrame to a new CSV file
df_cleaned.to_csv('cleaned_dataset_missing_data_handled.csv', index=False)

# Heatmap to visualize the cleaned data set
sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
plt.show()

# Visualize the data before handling outliers
sns.pairplot(df_cleaned)
plt.show()

### PART 5 : Handling Outliers ###

# Define a function to detect and handle outliers in df_cleaned based on the IQR method
def handle_outliers(columns):
    global df_cleaned  # Explicitly use df_cleaned in the function
    for col in columns:
        # Calculate the IQR for each column
        Q1 = df_cleaned[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df_cleaned[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        
        # Define the outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap the outliers using .loc to avoid SettingWithCopyWarning
        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound

# Select only the numeric columns in the dataset
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

# Apply outlier handling to numeric columns of df_cleaned
handle_outliers(numeric_cols)

# Display the first few rows of the outlier-handled dataset
print(df_cleaned.head())

# Save the dataset with handled outliers to a new CSV file
df_cleaned.to_csv('cleaned_dataset_outliers_handled.csv', index=False)

# Visualize the cleaned data after handling outliers
sns.pairplot(df_cleaned)
plt.show()
