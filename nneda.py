import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

# 1. Load the dataset
data = loadmat('traffic_dataset.mat')
print("Keys in the loaded .mat file:", data.keys())

# 2. Extract variables
X_train = data['tra_X_tr']
X_test = data['tra_X_te']
Y_train = data['tra_Y_tr']  # shape: (36, number_of_time_steps)
Y_test = data['tra_Y_te']
adj_mat = data['tra_adj_mat']  # shape: (36, 36)

# 3. Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
print("Adjacency matrix shape:", adj_mat.shape)

# 4. Inspect a sample from X_train and Y_train
print("\nFirst X_train sample (sparse matrix for one sequence):\n", X_train[0][0])
print("\nFirst Y_train target (all 36 sensors at first time step):\n", Y_train[:, 0])

# 5. Convert output arrays to DataFrames for easier manipulation
df_Y_train = pd.DataFrame(Y_train.T)  # shape: (time_steps, 36 sensors)
df_Y_test = pd.DataFrame(Y_test.T)
df_adj = pd.DataFrame(adj_mat)

# 6. Basic sensor statistics
# Mean and standard deviation per sensor (over time)
mean_per_sensor = df_Y_train.mean()
std_per_sensor = df_Y_train.std()

print("\nMean traffic volume per sensor (descending):")
print(mean_per_sensor.sort_values(ascending=False))

print("\nStd. dev. of traffic volume per sensor (descending):")
print(std_per_sensor.sort_values(ascending=False))

# 7. Correlation between sensors
corr_matrix = df_Y_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix Between Sensors (Traffic Patterns)")
plt.show()

# 8. Visualize traffic over time for a chosen sensor
sensor_id = 0  # example: sensor 0
plt.figure(figsize=(12, 4))
plt.plot(Y_train[sensor_id, :], label=f'Sensor {sensor_id}')
plt.title("Traffic Flow Over Time (Sensor 0)")
plt.xlabel("Time Step")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()

# 9. Basic stats across sensors
mean_traffic = np.mean(Y_train, axis=1)  # shape (36,)
max_traffic = np.max(Y_train, axis=1)
min_traffic = np.min(Y_train, axis=1)

# 10. Histogram of average traffic per sensor
plt.hist(mean_traffic, bins=30)
plt.title("Average Traffic per Sensor")
plt.xlabel("Average Volume")
plt.ylabel("Number of Sensors")
plt.show()

# 11. Check how many zero-traffic entries each sensor has
zero_counts = np.sum(Y_train == 0, axis=1)
plt.hist(zero_counts, bins=30)
plt.title("Zero Traffic Values per Sensor")
plt.xlabel("Number of Zeros")
plt.ylabel("Count of Sensors")
plt.show()

# 12. Visualize the adjacency matrix
plt.imshow(adj_mat, cmap='viridis')
plt.colorbar()
plt.title("Sensor Adjacency Matrix")
plt.xlabel("Sensor ID")
plt.ylabel("Sensor ID")
plt.show()

# 13. Convert a sparse input matrix to dense and visualize
#    X_train[0][0] is one sequence of input features for all sensors.
X_sample_sparse = X_train[0][0]
X_sample = X_sample_sparse.toarray() if hasattr(X_sample_sparse, 'toarray') else X_sample_sparse

plt.figure(figsize=(10, 4))
plt.imshow(X_sample, aspect='auto', cmap='hot')
plt.title("Sample Input Traffic Features (Sensor x Time)")
plt.xlabel("Time Step")
plt.ylabel("Sensor")
plt.colorbar(label="Traffic Volume")
plt.show()

# 14. (Optional) If the dataset encodes features in columns, define feature groups
feature_groups = {
    "Historical Traffic Volume (10)": list(range(0, 10)),
    "Weekday (7)": list(range(10, 17)),
    "Hour of Day (24)": list(range(17, 41)),
    "Road Direction (4)": list(range(41, 45)),
    "Number of Lanes (1)": [45],
    "Road Name (1)": [46]
}

# 15. Inspect the 'road name' feature in the first sample
df_X_sample = pd.DataFrame(X_sample)
road_name_col = df_X_sample[46]
unique_road_names = road_name_col.unique()
road_name_counts = road_name_col.value_counts().reset_index()
road_name_counts.columns = ['Encoded Road Name', 'Sensor Count']

print("\nUnique road name encodings in the first sample:\n", unique_road_names)
print("\nCount of each encoded road name:\n", road_name_counts)

# 16. (Optional) Save DataFrames to CSV
# df_Y_train.to_csv('Y_train.csv', index=False)
# df_Y_test.to_csv('Y_test.csv', index=False)
# df_adj.to_csv('adjacency_matrix.csv', index=False)
# df_X_sample.to_csv('X_sample_seq_input.csv', index=False)

print("\n--- EDA Complete ---")
# Continue with in-depth analysis of each feature group based on X_sample (first input sequence)
# We'll use the same df_X_sample DataFrame from before

feature_groups = {
    "Historical Traffic Volume": list(range(0, 10)),
    "Weekday Encoding": list(range(10, 17)),
    "Hour of Day Encoding": list(range(17, 41)),
    "Road Direction": list(range(41, 45)),
    "Number of Lanes": [45],
    "Road Name": [46]
}

# Summary containers
deep_feature_summary = {}

# Historical Traffic Volume (analyze mean & std for each time lag)
df_hist = df_X_sample[feature_groups["Historical Traffic Volume"]]
deep_feature_summary["Historical Traffic Volume Summary"] = df_hist.describe().T

# Weekday Encoding – One-hot encoded: check active weekday per sensor
weekday_active = df_X_sample[feature_groups["Weekday Encoding"]].idxmax(axis=1)
weekday_counts = weekday_active.value_counts().sort_index().reset_index()
weekday_counts.columns = ["Encoded Weekday Index", "Sensor Count"]
deep_feature_summary["Weekday Distribution"] = weekday_counts

# Hour of Day Encoding – One-hot encoded: count active hour feature per sensor
hour_active = df_X_sample[feature_groups["Hour of Day Encoding"]].idxmax(axis=1)
hour_counts = hour_active.value_counts().sort_index().reset_index()
hour_counts.columns = ["Encoded Hour Index", "Sensor Count"]
deep_feature_summary["Hour of Day Distribution"] = hour_counts

# Road Direction – already analyzed earlier, but repeat here for completeness
direction_active = df_X_sample[feature_groups["Road Direction"]].idxmax(axis=1)
direction_counts = direction_active.value_counts().sort_index().reset_index()
direction_counts.columns = ["Encoded Direction Index", "Sensor Count"]
deep_feature_summary["Direction Distribution"] = direction_counts

# Number of Lanes – value distribution
lanes_counts = df_X_sample[feature_groups["Number of Lanes"][0]].value_counts().reset_index()
lanes_counts.columns = ["Number of Lanes", "Sensor Count"]
deep_feature_summary["Lanes Distribution"] = lanes_counts

# Road Name – value distribution
road_counts = df_X_sample[feature_groups["Road Name"][0]].value_counts().reset_index()
road_counts.columns = ["Encoded Road Name", "Sensor Count"]
deep_feature_summary["Road Name Distribution"] = road_counts

# Show results
import ace_tools as tools
tools.display_dataframe_to_user(name="Deep Feature Analysis Summary", dataframe=pd.concat(deep_feature_summary.values(), keys=deep_feature_summary.keys()))
