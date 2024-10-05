import pandas as pd
import numpy as np

def detect_seismic_signals(csv_file, sta_length, lta_length, threshold):
    # Load data
    data = pd.read_csv(csv_file)
    
    # Extract velocity data
    print('getting velocity')
    print(data['velocity'])
    velocity = data['velocity'].values
    rel_time = data['rel_time(sec)'].values
    print(f'velocity: {velocity}, rel_time: {rel_time}')

    # Convert lengths from seconds to samples
    sampling_rate = 1  # Adjust according to your data's sampling rate
    sta_samples = int(sta_length * sampling_rate)
    lta_samples = int(lta_length * sampling_rate)
    
    # Initialize STA and LTA arrays
    sta = np.zeros_like(velocity)
    lta = np.zeros_like(velocity)
    signal_detected = np.zeros_like(velocity)
    
    # Calculate STA
    for i in range(sta_samples, len(velocity)):
        sta[i] = np.mean(velocity[i-sta_samples:i])
    
    # Calculate LTA
    for i in range(lta_samples + sta_samples, len(velocity)):
        lta[i] = np.mean(velocity[i-lta_samples:i])
    
    # Calculate the STA/LTA ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(lta != 0, sta / lta, 0)
    
    # Detect signals based on the threshold
    signal_detected[ratio > threshold] = 1
    
    # Extract times of detected signals
    detection_times = rel_time[signal_detected == 1]
    
    return detection_times
