import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Define folder paths
BASE_PATH = "C:\Users\user\Downloads\space_apps_2024_seismic_detection"
LUNAR_TRAINING_PATH = os.path.join(BASE_PATH, "data", "lunar", "training")
LUNAR_TEST_PATH = os.path.join(BASE_PATH, "data", "lunar", "test")
MARS_TRAINING_PATH = os.path.join(BASE_PATH, "data", "mars", "training")
MARS_TEST_PATH = os.path.join(BASE_PATH, "data", "mars", "test")

# Helper function to load time-series data (CSV files)
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    # Normalize the velocity column
    scaler = StandardScaler()
    if 'velocity(m/s)' in df.columns:
        df['velocity_normalized'] = scaler.fit_transform(df[['velocity(m/s)']])
    else:
        df['velocity_normalized'] = scaler.fit_transform(df[df.columns[1:2]])  # Assume second column is velocity
    # Reshape for LSTM input
    time_series_data = df['velocity_normalized'].values.reshape(-1, 1)
    return time_series_data

# Helper function to load images (plots)
def load_image(file_path, target_size=(224, 224)):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

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

# Load datasets (images, csvs, and labels) from directories
def load_datasets(data_path, catalog_path):
    image_data = []
    csv_data = []
    labels = []
    
    # Load the catalog for labeling
    catalog = pd.concat([pd.read_csv(os.path.join(catalog_path, file)) for file in os.listdir(catalog_path)], ignore_index=True)

    # Iterate over catalog and load corresponding data
    for _, row in catalog.iterrows():
        filename = row['filename']
        label = 1 if row['mq_type'] == 'impact_mq' else 0  # Label based on mq_type

        # Load image
        image_file = filename.replace('.csv', '.png')  # Assume the image has the same base name as CSV
        image_path = os.path.join(data_path, 'plots', 'images', image_file)
        if os.path.exists(image_path):
            img = load_image(image_path)
            image_data.append(img)

        # Load CSV (time-series data)
        csv_file = filename
        csv_path = os.path.join(data_path, 'data', 'S12_GradeB', 'csvs', csv_file)  # Adjust as per GradeA or GradeB
        if os.path.exists(csv_path):
            time_series = load_csv_data(csv_path)
            csv_data.append(time_series)
            labels.append(label)

    # Convert to numpy arrays
    image_data = np.array(image_data)
    csv_data = np.array(csv_data)
    labels = np.array(labels)
    
    return image_data, csv_data, labels

# Main function
def main():
    # Load lunar training datasets
    train_image_data, train_csv_data, train_labels = load_datasets(LUNAR_TRAINING_PATH, os.path.join(LUNAR_TRAINING_PATH, 'catalogs', 'csvs'))

    # Split the training data into training and validation sets
    X_train_img, X_val_img, X_train_csv, X_val_csv, y_train, y_val = train_test_split(train_image_data, train_csv_data, train_labels, test_size=0.2, random_state=42)

    # CNN input shape: (224, 224, 3) for images
    cnn_input_shape = (224, 224, 3)
    # LSTM input shape: based on the CSV data shape
    lstm_input_shape = (train_csv_data.shape[1], 1)

    # Create and compile the model
    model = create_combined_model(cnn_input_shape, lstm_input_shape)

    # Train the model
    model.fit([X_train_img, X_train_csv], y_train, epochs=10, batch_size=32, validation_data=([X_val_img, X_val_csv], y_val))

    # Save the model for later use
    model.save("seismic_model.h5")

    # Load lunar test datasets
    test_image_data, test_csv_data, test_labels = load_datasets(LUNAR_TEST_PATH, os.path.join(LUNAR_TEST_PATH, 'catalogs', 'csvs'))

    # Evaluate the model on test data
    loss, accuracy = model.evaluate([test_image_data, test_csv_data], test_labels)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
