# from seismic import detect_seismic_signals

# def main():
#     # Define parameters
#     csv_file = r'.\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-01-19HR00_evid00002.csv'
#     sta_length = 1.0  # Short-term average length in seconds
#     lta_length = 10.0  # Long-term average length in seconds
#     threshold = 2.0  # Threshold for detection

#     try:
#         detection_times = detect_seismic_signals(csv_file, sta_length, lta_length, threshold)
#         if len(detection_times) > 0:
#             print("Detected seismic signals at times (sec):", detection_times)
#         else:
#             print("No seismic signals detected.")
#     except FileNotFoundError:
#         print(f"Error: The file {csv_file} was not found.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#   main()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seismic import detect_seismic_signals

def main():
    # Define parameters
    csv_file = r'.\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-01-19HR00_evid00002.csv'
    sta_length = 1.0  # Short-term average length in seconds
    lta_length = 10.0  # Long-term average length in seconds
    threshold = 2.0  # Threshold for detection

    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(csv_file)

        # Assuming the data contains a column 'time' for timestamps and 'value' for the signal values
        time_column = 'time'  # Update this with the actual time column name if different
        value_column = 'value'  # Update this with the actual value column name if different
        
        # Detect seismic signals
        detection_times = detect_seismic_signals(csv_file, sta_length, lta_length, threshold)
        if len(detection_times) > 0:
            print("Detected seismic signals at times (sec):", detection_times)
        else:
            print("No seismic signals detected.")

        # Plot the data using Seaborn
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data[time_column], y=data[value_column], label='Signal', color='blue')

        # Mark detected seismic signals
        for detection_time in detection_times:
            plt.axvline(x=detection_time, color='red', linestyle='--', label='Detected Signal' if detection_time == detection_times[0] else "")

        plt.title('Seismic Signal Detection')
        plt.xlabel('Time (sec)')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.grid()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

