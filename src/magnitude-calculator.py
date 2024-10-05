import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, concatenate, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from obspy import read

# Helper function to load time-series data (CSV files) with truncation
def loadcsv_data(file_path, max_time_steps=1000):
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    if 'velocity(m/s)' in df.columns:
        df['velocity_normalized'] = scaler.fit_transform(df[['velocity(m/s)']])
    else:
        df['velocity_normalized'] = scaler.fit_transform(df[df.columns[1:2]])  # Assume second column is velocity

    # Limit to max_time_steps
    time_series_data = df['velocity_normalized'].values[:max_time_steps].reshape(-1, 1)
    return time_series_data

# Helper function to load images (plots)
def load_image(file_path, target_size=(224, 224)):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Helper function to load MSEED data
def load_mseed_data(mseed_file_path, max_time_steps=1000):
    try:
        st = read(mseed_file_path)  # Load the MSEED file
        tr = st[0]  # Assuming we're working with the first trace (for simplicity)
        data = tr.data[:max_time_steps]  # Truncate the data to max_time_steps
        data = data / np.max(np.abs(data))  # Normalize the waveform data
        return data.reshape(-1, 1)  # Return as 2D array (for consistency with CSV)
    except Exception as e:
        print(f"Error loading MSEED file {mseed_file_path}: {e}")
        return None

# Function to recursively find CSV files in all subfolders
def find_csv_in_subfolders(root_folder):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

# Function to remove .csv or .mseed extension if present and append .csv
def ensure_csv_extension(filename):
    # Check if it ends with .csv or .mseed and remove those specific extensions
    if filename.endswith('.csv') or filename.endswith('.mseed'):
        filename = filename.rsplit('.', 1)[0]  # Remove the last extension part (.csv or .mseed)
    
    # Append .csv extension
    return filename + '.csv'

# Function to load datasets with the actual folder structure and limited time steps for CSV
def load_datasets_with_mseed(data_path, catalog_path, csv_limit=None, limit=None, max_time_steps=1000):
    image_data, csv_data, mseed_data, labels = [], [], [], []
    csv_count = 0

    # Check if the catalog path exists
    if not os.path.exists(catalog_path):
        print(f"Catalog path does not exist: {catalog_path}")
        return None, None, None, None

    # Check if the plots folder exists where images are stored
    plots_path = os.path.join(data_path, 'plots')
    if not os.path.exists(plots_path):
        print(f"Plots directory does not exist: {plots_path}")
        return None, None, None, None

    # Recursively search for CSV files in the 'data' subfolders
    csv_files = find_csv_in_subfolders(os.path.join(data_path, 'data'))
    if not csv_files:
        print(f"No CSV files found in {os.path.join(data_path, 'data')}")
        return None, None, None, None

    # Load the catalog files
    catalog_files = [file for file in os.listdir(catalog_path) if file.endswith('.csv')]
    if not catalog_files:
        print(f"No catalog CSV files found in {catalog_path}")
        return None, None, None, None

    catalog = pd.concat([pd.read_csv(os.path.join(catalog_path, file)) for file in catalog_files], ignore_index=True)

    for i, row in catalog.iterrows():
        if limit and i >= limit:
            break

        filename = ensure_csv_extension(row['filename']) # Append .csv extension

        # Extract the numeric part of the 'evid' column and use it as the label
        evid_numeric = int(row['evid'].replace('evid', ''))
        label = 1 if evid_numeric % 2 == 0 else 0  # Example: label based on whether 'evid' is even or odd

        # Load the image from the 'plots' folder
        image_file = filename.replace('.csv', '.png')
        image_path = os.path.join(plots_path, image_file)
        if os.path.exists(image_path):
            img = load_image(image_path)
        else:
            print(f"Image file not found: {image_path}")
            continue  # Skip this sample if the image is missing

        # Find the CSV file in any of the subfolders within 'data'
        csv_file_path = None
        for csv_file in csv_files:
            if os.path.basename(csv_file) == filename:
                csv_file_path = csv_file
                break

        if csv_file_path:
            if csv_limit and csv_count >= csv_limit:
                print(f"Reached CSV limit of {csv_limit}. Skipping additional CSV files.")
                break
            time_series = load_csv_data(csv_file_path, max_time_steps)  # Truncate time series to max_time_steps
            csv_count += 1
        else:
            print(f"CSV file not found for: {filename}")
            continue  # Skip this sample if the CSV file is missing

        # Load the corresponding MSEED file (assumed to have the same base filename)
        mseed_file_path = csv_file_path.replace('.csv', '.mseed')
        if os.path.exists(mseed_file_path):
            mseed_series = load_mseed_data(mseed_file_path, max_time_steps)
            if mseed_series is None:
                continue  # Skip if MSEED file couldn't be loaded
        else:
            print(f"MSEED file not found: {mseed_file_path}")
            continue

        # Add the data to the lists
        image_data.append(img)
        csv_data.append(time_series)
        mseed_data.append(mseed_series)
        labels.append(label)

    if not image_data or not csv_data or not labels:
        raise ValueError("No data was loaded. Please check the file paths and ensure the dataset is properly formatted.")

    # Ensure consistent lengths across all datasets
    min_length = min(len(image_data), len(csv_data), len(mseed_data), len(labels))

    print(f"Number of samples: images={len(image_data)}, csvs={len(csv_data)}, mseeds={len(mseed_data)}, labels={len(labels)}")
    
    # Keep only the first `min_length` samples from each list to ensure consistency
    image_data = image_data[:min_length]
    csv_data = csv_data[:min_length]
    mseed_data = mseed_data[:min_length]
    labels = labels[:min_length]

    # Pad the time-series data so that all sequences have the same length
    csv_data_padded = pad_sequences(csv_data, padding='post', dtype='float32')
    mseed_data_padded = pad_sequences(mseed_data, padding='post', dtype='float32')

    return np.array(image_data), np.array(csv_data_padded), np.array(mseed_data_padded), np.array(labels)

# CNN model for image data
def create_cnn(input_shape):
    cnn_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    cnn_output = Dense(64, activation='relu')(x)
    
    return Model(inputs=cnn_input, outputs=cnn_output)

# LSTM model for time-series (CSV/MSEED) data
def create_lstm(input_shape):
    lstm_input = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(lstm_input)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    lstm_output = Dense(64, activation='relu')(x)
    
    return Model(inputs=lstm_input, outputs=lstm_output)

# Function to create a combined model with image, CSV, and MSEED inputs
def create_combined_model(cnn_input_shape, lstm_input_shape, mseed_input_shape):
    cnn_model = create_cnn(cnn_input_shape)
    lstm_csv_model = create_lstm(lstm_input_shape)  # For CSV data
    lstm_mseed_model = create_lstm(mseed_input_shape)  # For MSEED data

    # Combine CNN, CSV LSTM, and MSEED LSTM models
    combined = concatenate([cnn_model.output, lstm_csv_model.output, lstm_mseed_model.output])

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = Dropout(0.5)(x)
    final_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[cnn_model.input, lstm_csv_model.input, lstm_mseed_model.input], outputs=final_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to load lunar test datasets (without images, only CSV and MSEED data)
def load_lunar_test_data(test_data_path, max_time_steps=1000):
    csv_data, mseed_data = [], []
    
    # Recursively search for CSV files in the 'data' subfolders
    csv_files = find_csv_in_subfolders(os.path.join(test_data_path, 'data'))
    if not csv_files:
        print(f"No CSV files found in {os.path.join(test_data_path, 'data')}")
        return None, None

    for csv_file_path in csv_files:
        # Load the CSV data
        time_series = load_csv_data(csv_file_path, max_time_steps)

        # Load the corresponding MSEED file (assumed to have the same base filename)
        mseed_file_path = csv_file_path.replace('.csv', '.mseed')
        if os.path.exists(mseed_file_path):
            mseed_series = load_mseed_data(mseed_file_path, max_time_steps)
            if mseed_series is None:
                continue  # Skip if MSEED file couldn't be loaded
        else:
            print(f"MSEED file not found: {mseed_file_path}")
            continue

        # Add the data to the lists
        csv_data.append(time_series)
        mseed_data.append(mseed_series)

    if not csv_data or not mseed_data:
        print("No test data was loaded.")
        return None, None

    # Pad the time-series data so that all sequences have the same length
    csv_data_padded = pad_sequences(csv_data, padding='post', dtype='float32')
    mseed_data_padded = pad_sequences(mseed_data, padding='post', dtype='float32')

    return np.array(csv_data_padded), np.array(mseed_data_padded)

# Function to create an LSTM-only model for testing (since no images are available during test)
def create_lstm_only_model(lstm_input_shape, mseed_input_shape):
    lstm_csv_model = create_lstm(lstm_input_shape)  # For CSV data
    lstm_mseed_model = create_lstm(mseed_input_shape)  # For MSEED data

    # Combine the LSTM models (CSV and MSEED)
    combined = concatenate([lstm_csv_model.output, lstm_mseed_model.output])

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = Dropout(0.5)(x)
    final_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[lstm_csv_model.input, lstm_mseed_model.input], outputs=final_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_on_test_data():
    # Define the test data path
    LUNAR_TEST_PATH = os.path.join("data", "lunar", "test")

    # Load the lunar test dataset
    print("Loading lunar test data...")
    test_csv_data, test_mseed_data = load_lunar_test_data(LUNAR_TEST_PATH, max_time_steps=1000)

    if test_csv_data is None or test_mseed_data is None:
        print("Failed to load test data. Exiting.")
        return

    # CNN input shape: (224, 224, 3) for images (even if you won't use images during testing)
    cnn_input_shape = (224, 224, 3)
    # LSTM input shape: (1000, 1) for CSV and MSEED data
    lstm_input_shape = (1000, 1)
    mseed_input_shape = (1000, 1)

    # Create the full combined model (CNN + LSTM)
    combined_model = create_combined_model(cnn_input_shape, lstm_input_shape, mseed_input_shape)

    # Load the saved combined model's weights
    combined_model.load_weights('combined_model_mseed_integration.weights.h5')

    # Predict on the test data (just pass zeros for images)
    dummy_images = np.zeros((test_csv_data.shape[0], 224, 224, 3))  # Dummy image data
    predictions = combined_model.predict([dummy_images, test_csv_data, test_mseed_data], verbose=1)

    # Display the predictions
    for i, prediction in enumerate(predictions):
        print(f"Test Sample {i + 1}: Predicted Value: {prediction[0]}")

# Main function to load data, train, and test the model
def main():
    # Define folder paths for lunar data only
    LUNAR_TRAINING_PATH = os.path.join("data", "lunar", "training")

    # Load lunar training datasets with CSV and MSEED data
    print("Loading lunar data...")
    lunar_image_data, lunar_csv_data, lunar_mseed_data, lunar_labels = load_datasets_with_mseed(
        LUNAR_TRAINING_PATH, 
        os.path.join(LUNAR_TRAINING_PATH, 'catalogs'), 
        csv_limit=100,  # Increased to 100 CSV files for more data
        max_time_steps=1000
    )

    if lunar_image_data is None or lunar_csv_data is None or lunar_labels is None:
        print("Failed to load training data. Exiting.")
        return

    # Split the lunar data into training and validation sets
    X_train_img, X_val_img, X_train_csv, X_val_csv, X_train_mseed, X_val_mseed, y_train, y_val = train_test_split(
        lunar_image_data, lunar_csv_data, lunar_mseed_data, lunar_labels, test_size=0.2, random_state=42)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)

    # CNN input shape: (224, 224, 3) for images
    cnn_input_shape = (224, 224, 3)
    # LSTM input shape: (1000, 1) for truncated time-series data (CSV)
    lstm_input_shape = (1000, 1)
    # LSTM input shape: (1000, 1) for truncated MSEED time-series data
    mseed_input_shape = (1000, 1)

    # Create and compile the combined model
    combined_model = create_combined_model(cnn_input_shape, lstm_input_shape, mseed_input_shape)

    # Learning rate scheduler and early stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the combined model
    combined_model.fit(
        [X_train_img, X_train_csv, X_train_mseed], y_train,
        epochs=50,  # Increased epochs for better training
        batch_size=16,  # Reduced batch size
        validation_data=([X_val_img, X_val_csv, X_val_mseed], y_val),
        class_weight=class_weights_dict,
        callbacks=[lr_scheduler, early_stopping]
    )

    # Save the entire combined model's weights
    combined_model.save_weights('combined_model_mseed_integration.weights.h5')
    print("Combined model weights saved.")

    # Evaluate the model on lunar test data
    evaluate_on_test_data()

if __name__ == "__main__":
    main()

