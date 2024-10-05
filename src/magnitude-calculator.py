import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load seismic data from CSV file
data = pd.read_csv('C:\Users\user\Downloads\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-01-19HR00_evid00002.csv')

# Convert data to numpy arrays for easier processing
time_rel = data['time_rel(sec)'].values
velocity = data['velocity(m/s)'].values

# Function to compute STA/LTA ratio
def compute_sta_lta(velocity, sta_window, lta_window):
    sta = np.cumsum(velocity ** 2)
    lta = np.cumsum(velocity ** 2)

    # Calculate STA (Short-Term Average)
    sta = (sta[sta_window:] - sta[:-sta_window]) / sta_window

    # Calculate LTA (Long-Term Average)
    lta = (lta[lta_window:] - lta[:-lta_window]) / lta_window

    # Make STA and LTA of same length by truncating
    ratio = sta[lta_window - sta_window:] / lta
    
    # Pad with zeros at the beginning to match the original length
    ratio = np.concatenate([np.zeros(lta_window), ratio])

    return ratio

# Define STA and LTA window lengths
sta_window = 100   # Short-term window in data points
lta_window = 1000  # Long-term window in data points

# Compute STA/LTA ratio
sta_lta_ratio = compute_sta_lta(velocity, sta_window, lta_window)

# Define a threshold for detecting seismic events
threshold = 3.0

# Detect seismic events where the STA/LTA ratio exceeds the threshold
events = np.where(sta_lta_ratio > threshold)[0]

# Plot the results
plt.figure(figsize=(12, 6))

# Plot velocity data
plt.subplot(2, 1, 1)
plt.plot(time_rel, velocity, label='Velocity')
plt.title('Seismic Velocity Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')

# Plot STA/LTA ratio
plt.subplot(2, 1, 2)
plt.plot(time_rel, sta_lta_ratio, label='STA/LTA Ratio', color='orange')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('STA/LTA Ratio and Detected Events')
plt.xlabel('Time (seconds)')
plt.ylabel('STA/LTA Ratio')

# Mark detected events
for event in events:
    plt.axvline(x=time_rel[event], color='green', linestyle='--')

plt.legend()
plt.tight_layout()
plt.show()
