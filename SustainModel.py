import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

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

print('### Data Frame Visualization ###')

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
# print(df_merged.sample(n=10))

### PART 4 : Handling Missing Data ###

# Heatmap to visualize missing data
print('### Heatmap to visualize missing data ###')
sns.heatmap(df_merged.isnull(), cbar=False, cmap='viridis')
plt.show()

# Filtering the merged data set to keep the cleaned data 

# Remove rows from '2006-10-02' onwards
df_cleaned = df_merged[df_merged['datetime'] > '2006-10-02']

# Display the filtered DataFrame
# print(df_cleaned.head())

# Save the filtered DataFrame to a new CSV file
df_cleaned.to_csv('cleaned_dataset_missing_data_handled.csv', index=False)

# Heatmap to visualize the cleaned data set
print('### Heatmap to visualize the cleaned data set ###')
sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
plt.show()

# Visualize the data before handling outliers
print('### Visualize the data before handling outliers ###')
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
# print(df_cleaned.head())

# Save the dataset with handled outliers to a new CSV file
df_cleaned.to_csv('cleaned_dataset_outliers_handled.csv', index=False)

# Visualize the cleaned data after handling outliers
print('### Visualize the data after handling outliers ###')
sns.pairplot(df_cleaned)
plt.show()

### PART 6 : Ensuring Correct Data Types ###
df_cleaned = pd.read_csv('cleaned_dataset_outliers_handled.csv')

# print(df_cleaned.dtypes)

# Ensure 'datetime' column is in datetime format
df_cleaned['datetime'] = pd.to_datetime(df_cleaned['datetime'], errors='coerce')
is_datetime = pd.api.types.is_datetime64_any_dtype(df_cleaned['datetime'])
# print(f"\nIs 'datetime' column in datetime format? {is_datetime}")

cols = ['area', 'storage', 'delta storage', 'inflow', 'inflow volume', 'unregulated inflow', 'unregulated inflow volume', 'mod unregulated inflow', 'mod unregulated inflow volume', 'evaporation', 'release volume', 'total release']

for column in cols:
    # Check if each value in the column is a float and print the result
    is_float_column = df_cleaned[column].apply(lambda x: isinstance(x, float))
    # print(f"Are all values in '{column}' float? {is_float_column.all()}")

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
# print(df_new_feature[['datetime', 'net_flow_volume', 'net_storage_volume']].head())

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
# print(monthly_avg.head())

# Create a dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot for 'net flow volume' on the first y-axis
ax1.plot(monthly_avg['datetime'], monthly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Net Flow Volume', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Monthly Average of Net Flow Volume and Net Storage Volume Over Time')
ax1.grid(True)

# Create a second y-axis for 'net storage volume'
ax2 = ax1.twinx()
ax2.plot(monthly_avg['datetime'], monthly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
ax2.set_ylabel('Net Storage Volume', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Half yearly average of 'net flow volume' and 'net storage volume'

# Resample the data to 6-month (half-yearly) frequency and calculate the mean
half_yearly_avg = df_new_feature.set_index('datetime').resample('6M').mean()

# Reset index to get 'datetime' back as a column for plotting
half_yearly_avg.reset_index(inplace=True)

# Display the first few rows of the half-yearly averaged data
# print(half_yearly_avg.head())

# Create a dual-axis plot for half-yearly averages
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot for 'net flow volume' on the first y-axis
ax1.plot(half_yearly_avg['datetime'], half_yearly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Net Flow Volume', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Half-Yearly Average of Net Flow Volume and Net Storage Volume Over Time')
ax1.grid(True)

# Create a second y-axis for 'net storage volume'
ax2 = ax1.twinx()
ax2.plot(half_yearly_avg['datetime'], half_yearly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
ax2.set_ylabel('Net Storage Volume', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Yearly average of 'net flow volume' and 'net storage volume'

# Resample the data to yearly frequency and calculate the mean
yearly_avg = df_new_feature.set_index('datetime').resample('Y').mean()

# Reset index to get 'datetime' back as a column for plotting
yearly_avg.reset_index(inplace=True)

# Display the first few rows of the yearly averaged data
# print(yearly_avg.head())

# Create a dual-axis plot for yearly averages
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot for 'net flow volume' on the first y-axis
ax1.plot(yearly_avg['datetime'], yearly_avg['net_flow_volume'], label='Net Flow Volume', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Net Flow Volume', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Yearly Average of Net Flow Volume and Net Storage Volume Over Time')
ax1.grid(True)

# Create a second y-axis for 'net storage volume'
ax2 = ax1.twinx()
ax2.plot(yearly_avg['datetime'], yearly_avg['net_storage_volume'], label='Net Storage Volume', color='g')
ax2.set_ylabel('Net Storage Volume', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### PART 8 : Model Selection, Trainig, and Evaluation ###

### Linear Regression Model ###

# Convert 'datetime' into numerical format
df_new_feature['datetime_num'] = pd.to_datetime(df_new_feature['datetime']).map(datetime.toordinal)

# Define X (datetime as numerical) and Y (net storage volume)
X = df_new_feature[['datetime_num']]
Y = df_new_feature['net_storage_volume']

# Create and train the linear regression model
model_storage = LinearRegression()
model_storage.fit(X, Y)

# Predict when 'net storage volume' would hit 0
# This is equivalent to solving for the 'datetime_num' where Y=0

predicted_datetime_num = -model_storage.intercept_ / model_storage.coef_[0]

# Convert predicted 'datetime_num' back to a readable date format
predicted_date = datetime.fromordinal(int(predicted_datetime_num))

# Print the predicted date
# print(f"The predicted date when net storage volume will hit 0 is: {predicted_date}")

# Plot the linear regression fit and the prediction
plt.figure(figsize=(10, 6))
plt.scatter(df_new_feature['datetime'], df_new_feature['net_storage_volume'], color='green', label='Data')
plt.plot(df_new_feature['datetime'], model_storage.predict(X), color='blue', label='Linear Fit')
plt.axhline(0, color='red', linestyle='--', label='Net Storage = 0')
plt.axvline(predicted_date, color='orange', linestyle='--', label=f'Prediction: {predicted_date.date()}')
plt.title('Prediction of When Net Storage Volume Hits 0')
plt.xlabel('Date')
plt.ylabel('Net Storage Volume')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define X (datetime as numerical) and Y (net flow volume)
X = df_new_feature[['datetime_num']]
Y = df_new_feature['net_flow_volume']

# Now, let's model the net flow volume based on datetime
model_flow = LinearRegression()
model_flow.fit(X, Y)

# Predict the net flow volume at the time when net storage volume hits 0
predicted_net_flow_at_zero_storage = model_flow.predict([[predicted_datetime_num]])

# Plot the results for net flow volume
plt.figure(figsize=(10, 6))

# Plot the net flow volume behavior as net storage volume hits 0
plt.scatter(df_new_feature['datetime'], df_new_feature['net_flow_volume'], color='blue', label='Net Flow Volume Data')
plt.plot(df_new_feature['datetime'], model_flow.predict(X), color='green', label='Net Flow Volume Linear Fit')
plt.axvline(predicted_date, color='orange', linestyle='--', label=f'Prediction: {predicted_date.date()}')
plt.axhline(predicted_net_flow_at_zero_storage[0], color='red', linestyle='--', label=f'Net Flow at 0 Storage: {predicted_net_flow_at_zero_storage[0]:.2f}')
plt.title('Behavior of Net Flow Volume as Net Storage Volume Hits 0')
plt.xlabel('Date')
plt.ylabel('Net Flow Volume')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

### PART 9 : Final Prediction ###

# Print the results
print(f"The predicted date when net storage volume will hit 0 is: {predicted_date}")
print(f"The predicted net flow volume at this point is: {predicted_net_flow_at_zero_storage[0]}")

### PART 10 : Solution Model ###

# Constants
TARGET_STORAGE = 1400000  # Target net storage volume (in whatever unit the dataset uses)
CURRENT_STORAGE = df['net_storage_volume'].iloc[-1]  # The last known net storage volume
YEARS_5 = 5
YEARS_10 = 10

# Define the timeframe for 5 and 10 years from the current date
current_date = df['datetime'].max()
# Convert current_date to a datetime object
current_date = pd.to_datetime(current_date) 
future_date_5_years = current_date + timedelta(days=YEARS_5 * 365)
future_date_10_years = current_date + timedelta(days=YEARS_10 * 365)

# Timeframe in days (to calculate required flow rate)
days_5_years = (future_date_5_years - current_date).days
days_10_years = (future_date_10_years - current_date).days

# Calculate the required 'net flow volume' rate

# Define the 'net flow volume' needed to reach target storage in 5 and 10 years
required_flow_rate_5_years = (TARGET_STORAGE - CURRENT_STORAGE) / days_5_years
required_flow_rate_10_years = (TARGET_STORAGE - CURRENT_STORAGE) / days_10_years

# Create future dates for 5 and 10 years
future_dates_5_years = pd.date_range(current_date, future_date_5_years - timedelta(days=1), freq='D')
future_dates_10_years = pd.date_range(current_date, future_date_10_years, freq='D')

# Predict future 'net storage volume' for 5 and 10 years

# Predict 'net storage volume' based on required flow rate over time for both 5 and 10 years
predicted_storage_5_years = [CURRENT_STORAGE + i * required_flow_rate_5_years for i in range(days_5_years)]
predicted_storage_10_years = [CURRENT_STORAGE + i * required_flow_rate_10_years for i in range(days_10_years)]

# Plot the future net flow volume and net storage volume for 5 years
plt.figure(figsize=(12, 12))

# Fetch the actual net flow volume and net storage volume from the dataset for plotting
actual_net_flow_volume = df['net_flow_volume']
actual_net_storage_volume = df['net_storage_volume']

# Ensure that 'datetime' column is in proper format
df['datetime'] = pd.to_datetime(df['datetime'])

# Plot for required flow rate
plt.subplot(3, 1, 1)
plt.plot(future_dates_5_years, [required_flow_rate_5_years] * len(future_dates_5_years), label='Required Average Flow Rate (5 Years)', color='blue')
plt.plot(future_dates_10_years, [required_flow_rate_10_years] * len(future_dates_10_years), label='Required Average Flow Rate (10 Years)', color='green')
plt.plot(df['datetime'], actual_net_flow_volume, label='Actual Net Flow Volume', color='orange')
plt.title('Net Flow Volume Rates')
plt.xlabel('Date')
plt.ylabel('Flow Volume Rate')
plt.legend()
plt.grid(True)

# Plot for predicted storage volume
plt.subplot(3, 1, 2)
plt.plot(future_dates_5_years, predicted_storage_5_years, label='Predicted Storage Volume (5 Years)', color='orange')
plt.plot(future_dates_10_years[:len(predicted_storage_10_years)], predicted_storage_10_years, label='Predicted Storage Volume (10 Years)', color='red')
plt.plot(df['datetime'], actual_net_storage_volume, label='Actual Net Storage Volume', color='purple')
plt.axhline(TARGET_STORAGE, color='purple', linestyle='--', label=f'Target Storage: {TARGET_STORAGE}')
plt.title('Predicted and Actual Net Storage Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Net Storage Volume')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

### PART 11 : Final Solution ###

# Print the results
print(f"Required net flow volume rate to reach {TARGET_STORAGE} in 5 years: {required_flow_rate_5_years} per day")
print(f"Required net flow volume rate to reach {TARGET_STORAGE} in 10 years: {required_flow_rate_10_years} per day")
