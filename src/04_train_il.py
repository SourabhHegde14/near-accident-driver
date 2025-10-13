# src/04_train_il.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import build_model
import os

# --- Configuration ---
DATA_DIR = '../data'
CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')
# UPDATED LINE: Changed file extension from .h5 to .keras
MODEL_PATH = '../saved_models/il_model.keras'
IMAGE_HEIGHT = 130
IMAGE_WIDTH = 600
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 15

def data_generator(data_frame, batch_size):
    """Generator to feed image data and actions to the Keras model."""
    num_samples = len(data_frame)
    while True:
        data_frame = data_frame.sample(frac=1) # Shuffle data
        for offset in range(0, num_samples, batch_size):
            batch_samples = data_frame.iloc[offset:offset+batch_size]
            
            images = []
            actions = []
            for index, row in batch_samples.iterrows():
                img_path = os.path.join(DATA_DIR, row['image_path'].strip())
                image = load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
                image = img_to_array(image)
                
                action = int(row['action'])
                
                images.append(image)
                actions.append(action)
            
            yield np.array(images), np.array(actions)

def main():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_gen = data_generator(train_df, BATCH_SIZE)
    val_gen = data_generator(val_df, BATCH_SIZE)
    
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    model = build_model(input_shape)
    model.summary()
    
    print("\n--- Starting Imitation Learning Training ---")
    model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # The model will now be saved in the modern .keras format
    model.save(MODEL_PATH)
    print(f"--- Model saved to {MODEL_PATH} ---")

if __name__ == '__main__':
    main()