import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data (time_rel and velocity columns)
df = pd.read_csv('/Users/tatiatsiklauri/Desktop/space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB/xa.s12.00.mhz.1969-12-16HR00_evid00006.csv')  # Load your data here

# STA/LTA Parameters
short_window = 50   # Short-term window size (in data points)
long_window = 500   # Long-term window size (in data points)
threshold_on = 4.0  # Detection threshold
threshold_off = 1.5 # End of event threshold

# Calculate the STA/LTA characteristic function
def classic_sta_lta(velocity, short_window, long_window):
    sta = np.cumsum(velocity ** 2)
    sta[short_window:] = sta[short_window:] - sta[:-short_window]
    lta = np.cumsum(velocity ** 2)
    lta[long_window:] = lta[long_window:] - lta[:-long_window]
    
    # Avoid division by zero
    lta[lta == 0] = 1e-10
    sta_lta_ratio = sta[long_window - 1:] / lta[long_window - 1:]
    
    return sta_lta_ratio

# Apply STA/LTA to the velocity data
velocity = df['velocity(m/s)'].values
sta_lta_ratio = classic_sta_lta(velocity, short_window, long_window)

# Detect seismic events based on the STA/LTA ratio
seismic_event_onset = np.where(sta_lta_ratio > threshold_on)[0]
seismic_event_offset = np.where(sta_lta_ratio < threshold_off)[0]

# Find and mark events (simple logic for this example)
events = []
for onset in seismic_event_onset:
    offset = seismic_event_offset[seismic_event_offset > onset][0]
    events.append((onset, offset))

# Plot results
plt.figure(figsize=(10, 6))

# Plot velocity
plt.subplot(2, 1, 1)
plt.plot(df['time_rel(sec)'], velocity, label='Velocity (m/s)')
plt.title('Seismic Event Detection using STA/LTA')
plt.xlabel('Relative Time (s)')
plt.ylabel('Velocity (m/s)')

# Mark detected events on velocity plot
for event in events:
    plt.axvspan(df['time_rel(sec)'][event[0]], df['time_rel(sec)'][event[1]], color='red', alpha=0.3)

# Plot STA/LTA characteristic function
plt.subplot(2, 1, 2)
plt.plot(df['time_rel(sec)'][long_window-1:], sta_lta_ratio, label='STA/LTA Ratio', color='g')
plt.axhline(y=threshold_on, color='r', linestyle='--', label='Threshold On')
plt.axhline(y=threshold_off, color='b', linestyle='--', label='Threshold Off')
plt.title('STA/LTA Characteristic Function')
plt.xlabel('Relative Time (s)')
plt.ylabel('STA/LTA Ratio')
plt.legend()

plt.tight_layout()
plt.show()

# Extract the maximum velocity from the detected seismic events
max_velocity = max(abs(df['velocity(m/s)']))  # Taking absolute value of velocity

# Assume a constant C based on empirical values (adjust as needed for accuracy)
C = 5.5  # This constant should be calibrated for your region/data

# Estimate magnitude
magnitude = np.log10(max_velocity) + C

print(f"Estimated Magnitude: {magnitude}")