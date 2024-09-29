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
# Plot scatter plots for all numerical columns in each dataframe against 'datetime'

# Convert 'datetime' column to datetime format in each DataFrame
for name, df in dataframes.items():
    # Check if 'datetime' column exists before converting to datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        print(f"Warning: 'datetime' column not found in {name} DataFrame")

# Plot scatter plots for all numerical columns in each DataFrame against 'datetime'
for name, df in dataframes.items():
    if 'datetime' in df.columns: 
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create scatter plots for each numeric column against 'datetime'
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['datetime'], df[col], alpha=0.6)
            plt.title(f'Scatter Plot of {col} vs Datetime in {name} Dataframe', fontsize=16)
            plt.xlabel('Datetime')
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        print(f"Skipping {name} DataFrame due to missing 'datetime' column")
    
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

# Display the random rows of the merged DataFrame
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

### PART 6 : Ensuring Correct Data Types ###
df_cleaned = pd.read_csv('cleaned_dataset_outliers_handled.csv')

print(df_cleaned.dtypes)

# Ensure 'datetime' column is in datetime format
df_cleaned['datetime'] = pd.to_datetime(df_cleaned['datetime'], errors='coerce')
is_datetime = pd.api.types.is_datetime64_any_dtype(df_cleaned['datetime'])
print(f"\nIs 'datetime' column in datetime format? {is_datetime}")

cols = ['area', 'storage', 'delta storage', 'inflow', 'inflow volume', 'unregulated inflow', 'unregulated inflow volume', 'mod unregulated inflow', 'mod unregulated inflow volume', 'evaporation', 'release volume', 'total release']

for column in cols:
    # Check if each value in the column is a float and print the result
    is_float_column = df_cleaned[column].apply(lambda x: isinstance(x, float))
    print(f"Are all values in '{column}' float? {is_float_column.all()}")

#check for duplicate values in each column
df_cleaned[df_cleaned[cols].duplicated()]

### PART 7 : Adding New Features ###

# Create a new DataFrame to hold the new features
df_new_feature = df_cleaned.copy()

# Calculate 'net flow volume'
df_new_feature['net_flow_volume'] = (
    df_new_feature['release volume'] - 
    df_new_feature['inflow volume'] + 
    df_new_feature['evaporation']
)

# Ensure 'storage' column exists for net storage calculation
# Create 'net storage volume', starting from 2006-10-03
df_new_feature['net_storage_volume'] = df_new_feature['storage'].shift(1) + df_new_feature['net_flow_volume']

# Drop the first row since it will have NaN values in 'net_storage_volume'
df_new_feature.dropna(inplace=True)

# Save the DataFrame with new features to a new CSV file
df_new_feature.to_csv('cleaned_dataset_with_new_features.csv', index=False)

# Display the first few rows of the DataFrame with new features
print(df_new_feature[['datetime', 'net_flow_volume', 'net_storage_volume']].head())

# Visualization 

# Scatter plot of 'net flow volume' and 'net storage volume' against 'datetime'
plt.figure(figsize=(12, 6))

# Scatter plot for net flow volume
plt.subplot(2, 1, 1)
plt.scatter(df_new_feature['datetime'], df_new_feature['net_flow_volume'], color='blue', alpha=0.6, label='Net Flow Volume')
plt.title('Net Flow Volume over Time')
plt.xlabel('Datetime')
plt.ylabel('Net Flow Volume')
plt.xticks(rotation=45)
plt.legend()

# Scatter plot for net storage volume
plt.subplot(2, 1, 2)
plt.scatter(df_new_feature['datetime'], df_new_feature['net_storage_volume'], color='green', alpha=0.6, label='Net Storage Volume')
plt.title('Net Storage Volume over Time')
plt.xlabel('Datetime')
plt.ylabel('Net Storage Volume')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Monthly average of 'net flow volume' and 'net storage volume'

# Filter the DataFrame to include data from 2006-10-03 onwards
df_filtered = df_new_feature[df_new_feature['datetime'] >= '2006-10-03']

# Set 'datetime' as the index
df_filtered.set_index('datetime', inplace=True)

# Resample the data to get monthly averages for 'net_flow_volume' and 'net_storage_volume'
monthly_avg = df_filtered[['net_flow_volume', 'net_storage_volume']].resample('M').mean()

# Reset index to get 'datetime' back as a column
monthly_avg.reset_index(inplace=True)

# Display the first few rows of the monthly averages DataFrame
print(monthly_avg.head())

# Plot for 'net flow volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg['datetime'], monthly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
plt.xlabel('Date')
plt.ylabel('Net Flow Volume')
plt.title('Monthly Average of Net Flow Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot for 'net storage volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg['datetime'], monthly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
plt.xlabel('Date')
plt.ylabel('Net Storage Volume')
plt.title('Monthly Average of Net Storage Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Half yearly average of 'net flow volume' and 'net storage volume'

# Resample the data to 6-month (half-yearly) frequency and calculate the mean
half_yearly_avg = df_new_feature.set_index('datetime').resample('6M').mean()

# Reset index to get 'datetime' back as a column for plotting
half_yearly_avg.reset_index(inplace=True)

# Display the first few rows of the half-yearly averaged data
print(half_yearly_avg.head())

# Plot for 'net flow volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(half_yearly_avg['datetime'], half_yearly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
plt.xlabel('Date')
plt.ylabel('Net Flow Volume')
plt.title('Half-Yearly Average of Net Flow Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot for 'net storage volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(half_yearly_avg['datetime'], half_yearly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
plt.xlabel('Date')
plt.ylabel('Net Storage Volume')
plt.title('Half-Yearly Average of Net Storage Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Yearly average of 'net flow volume' and 'net storage volume'

# Resample the data to yearly frequency and calculate the mean
yearly_avg = df_new_feature.set_index('datetime').resample('Y').mean()

# Reset index to get 'datetime' back as a column for plotting
yearly_avg.reset_index(inplace=True)

# Display the first few rows of the yearly averaged data
print(yearly_avg.head())

# Plot for 'net flow volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg['datetime'], yearly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
plt.xlabel('Date')
plt.ylabel('Net Flow Volume')
plt.title('Yearly Average of Net Flow Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot for 'net storage volume' against 'datetime'
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg['datetime'], yearly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
plt.xlabel('Date')
plt.ylabel('Net Storage Volume')
plt.title('Yearly Average of Net Storage Volume Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

### PART 8 : Model Selection ###

### PART 9 : Model Training ###

### PART 10 : Model Evaluation ###

### PART 11 : Prediction ###
