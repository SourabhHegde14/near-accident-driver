# src/03_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda

def build_model(input_shape):
    """Builds the CNN model for 5 discrete actions."""
    model = Sequential()
    
    # Pre-processing: Normalize the image data to be between -0.5 and 0.5
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    
    # Convolutional Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    model.add(Flatten())
    
    # Dense Layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    
    # Output layer: 5 actions, so 5 outputs with softmax for probability
    model.add(Dense(5, activation='softmax')) 
    
    # Compile the model for classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # Example: Height=130, Width=600, Channels=3 (BGR)
    INPUT_SHAPE = (130, 600, 3) 
    model = build_model(INPUT_SHAPE)
    model.summary()