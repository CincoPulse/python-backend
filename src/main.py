from seismic import detect_seismic_signals

def main():
    # Define parameters
    csv_file = r'.\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-01-19HR00_evid00002.csv'
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
<<<<<<< HEAD
  main()
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os



=======
    main()
>>>>>>> 7c280d1d0ae8c7f001125699171839f850795a87
