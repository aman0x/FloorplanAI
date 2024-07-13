# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess data from a directory
def load_data(directory, img_size, batch_size):
    # Initialize the data generator with rescaling and validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Generate batches of image data (and their labels) with real-time data augmentation
    train_gen = datagen.flow_from_directory(
        directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse',  # 'sparse' for integer labels
        subset='training'  # Set as training data
    )

    # Generate batches of image data (and their labels) for validation
    val_gen = datagen.flow_from_directory(
        directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse',  # 'sparse' for integer labels
        subset='validation'  # Set as validation data
    )

    return train_gen, val_gen
