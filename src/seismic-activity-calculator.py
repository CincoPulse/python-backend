import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define folder paths
LUNAR_TRAINING_PATH = os.path.join("data", "lunar", "training")
LUNAR_TEST_PATH = os.path.join("data", "lunar", "test")

# Helper function to load time-series data (CSV files) with truncation
def load_csv_data(file_path, max_time_steps=1000):
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

# Function to recursively find CSV files in all subfolders
def find_csv_in_subfolders(root_folder):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

# Function to load datasets with the actual folder structure and limited time steps for CSV
def load_datasets(data_path, catalog_path, csv_limit=None, limit=None, max_time_steps=1000):
    image_data, csv_data, labels = [], [], []
    csv_count = 0

    # Check if the catalog path exists
    if not os.path.exists(catalog_path):
        print(f"Catalog path does not exist: {catalog_path}")
        return None, None, None

    # Check if the plots folder exists where images are stored
    plots_path = os.path.join(data_path, 'plots')
    if not os.path.exists(plots_path):
        print(f"Plots directory does not exist: {plots_path}")
        return None, None, None

    # Recursively search for CSV files in the 'data' subfolders
    csv_files = find_csv_in_subfolders(os.path.join(data_path, 'data'))
    if not csv_files:
        print(f"No CSV files found in {os.path.join(data_path, 'data')}")
        return None, None, None

    # Load the catalog files
    catalog_files = [file for file in os.listdir(catalog_path) if file.endswith('.csv')]
    if not catalog_files:
        print(f"No catalog CSV files found in {catalog_path}")
        return None, None, None

    catalog = pd.concat([pd.read_csv(os.path.join(catalog_path, file)) for file in catalog_files], ignore_index=True)

    for i, row in catalog.iterrows():
        if limit and i >= limit:
            break

        filename = row['filename'] + '.csv'  # Append .csv extension
        label = 1 if row['mq_type'] == 'impact_mq' else 0

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

        # Add the data to the lists
        image_data.append(img)
        csv_data.append(time_series)
        labels.append(label)

    if not image_data or not csv_data or not labels:
        raise ValueError("No data was loaded. Please check the file paths and ensure the dataset is properly formatted.")

    # Ensure consistent lengths across all datasets
    min_length = min(len(image_data), len(csv_data), len(labels))

    print(f"Number of samples: images={len(image_data)}, csvs={len(csv_data)}, labels={len(labels)}")
    
    # Keep only the first `min_length` samples from each list to ensure consistency
    image_data = image_data[:min_length]
    csv_data = csv_data[:min_length]
    labels = labels[:min_length]

    # Pad the time-series data so that all sequences have the same length
    csv_data_padded = pad_sequences(csv_data, padding='post', dtype='float32')

    return np.array(image_data), np.array(csv_data_padded), np.array(labels)

# CNN model for image data
def create_cnn(input_shape):
    cnn_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    cnn_output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=cnn_input, outputs=cnn_output)

# LSTM model for time-series (CSV) data
def create_lstm(input_shape):
    lstm_input = Input(shape=input_shape)
    x = LSTM(50)(lstm_input)
    lstm_output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=lstm_input, outputs=lstm_output)

# Function to create a combined model
def create_combined_model(cnn_input_shape, lstm_input_shape):
    cnn_model = create_cnn(cnn_input_shape)
    lstm_model = create_lstm(lstm_input_shape)

    # Combine CNN and LSTM models
    combined = concatenate([cnn_model.output, lstm_model.output])

    final_output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=final_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Main function to load data and train the model
def main():
    # Define folder paths
    LUNAR_TRAINING_PATH = os.path.join("data", "lunar", "training")
    LUNAR_TEST_PATH = os.path.join("data", "lunar", "test")

    # Load lunar training datasets with limited time-series steps and csv limit
    train_image_data, train_csv_data, train_labels = load_datasets(
        LUNAR_TRAINING_PATH, 
        os.path.join(LUNAR_TRAINING_PATH, 'catalogs'), 
        csv_limit=50,  # Limit to 50 CSV files
        max_time_steps=1000  # Truncate time-series data to 1000 time steps
    )

    if train_image_data is None or train_csv_data is None or train_labels is None:
        print("Failed to load training data. Exiting.")
        return

    # Split the training data into training and validation sets
    X_train_img, X_val_img, X_train_csv, X_val_csv, y_train, y_val = train_test_split(
        train_image_data, train_csv_data, train_labels, test_size=0.2, random_state=42)

    # CNN input shape: (224, 224, 3) for images
    cnn_input_shape = (224, 224, 3)
    # LSTM input shape: (1000, 1) for truncated time-series data
    lstm_input_shape = (1000, 1)

    # Create and compile the model
    model = create_combined_model(cnn_input_shape, lstm_input_shape)

    # Train the model
    model.fit([X_train_img, X_train_csv], y_train, epochs=10, batch_size=32, validation_data=([X_val_img, X_val_csv], y_val))

    # Save the model in the new Keras format
    model.save("seismic_model.keras")

    # Load lunar test datasets
    test_image_data, test_csv_data, test_labels = load_datasets(
        LUNAR_TEST_PATH, 
        os.path.join(LUNAR_TEST_PATH, 'catalogs'),
        max_time_steps=1000  # Same truncation for test data
    )

    # Handle case when test data is not loaded properly
    if test_image_data is None or test_csv_data is None or test_labels is None:
        print("Failed to load test data. Exiting.")
        return

    # Evaluate the model on test data
    loss, accuracy = model.evaluate([test_image_data, test_csv_data], test_labels)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
