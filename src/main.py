import pandas as pd
import numpy as np

def detect_seismic_signals(csv_file, sta_length, lta_length, threshold):
    # Load data
    data = pd.read_csv(csv_file)
    
    # Extract velocity data
    velocity = data['velocity'].values
    rel_time = data['rel_time(sec)'].values
    
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

def main():
    # Define parameters
    csv_file = 'path/to/your/data.csv'
    sta_length = 1.0  # Short-term average length in seconds
    lta_length = 10.0  # Long-term average length in seconds
    threshold = 2.0  # Threshold for detection

    try:
        detection_times = detect_seismic_signals(csv_file, sta_length, lta_length, threshold)
        if len(detection_times) > 0:
            print("Detected seismic signals at times (sec):", detection_times)
        else:
            print("No seismic signals detected.")
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
