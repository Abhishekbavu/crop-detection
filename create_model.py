"""
Script to create a dummy disease detection model
This creates a simple CNN model for demonstration purposes
Run this script once to generate the disease_model.h5 file
"""
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    import os
    
    # Check if trained model already exists
    if os.path.exists('models/my_cnn_model.h5'):
        print("✓ Found existing trained model: models/my_cnn_model.h5")
        print("  The application is configured to use this model.")
        print("\nIf you want to create a dummy model instead, delete the existing one first.")
    else:
        print("Creating dummy disease detection model...")
        
        # Define a simple CNN model for plant disease detection
        # Matching the trained model architecture (150x150 input, 38 classes)
        model = keras.Sequential([
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(38, activation='softmax')  # 38 disease classes
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Save the model
        model.save('models/my_cnn_model.h5')
        print("\n✓ Model saved successfully to models/my_cnn_model.h5")
        print("  This is a dummy model for demonstration purposes.")
        print("  For production use, train the model with actual plant disease dataset.")
        print("\nRecommended datasets:")
        print("  - PlantVillage Dataset")
        print("  - PlantDoc Dataset")
        print("  - Kaggle Plant Disease Datasets")
    
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install TensorFlow first:")
    print("  pip install tensorflow")
    
except Exception as e:
    print(f"Error creating model: {e}")
