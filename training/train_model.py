from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
print(os.getcwd())

from model.cnn_model import create_model
from utils.data_loader import load_data


# Configuration settings (example values, adjust as needed)
img_size = 256  # Size of the input images
batch_size = 32  # Number of images in each batch
num_classes = 5  # Number of classes in your dataset
epochs = 20  # Number of epochs to train the model

def train_model():
    # Load training and validation data
    train_gen, val_gen = load_data('data/floorplans/', img_size, batch_size)

    # Create the model
    model = create_model((img_size, img_size, 3), num_classes)

    # Callbacks to save the model and early stopping
    checkpoint = ModelCheckpoint('model/floorplan_model_best.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop]
    )

    # Save the final model
    model.save('model/floorplan_model_best.keras')

    return history

if __name__ == '__main__':
    history = train_model()
    print("Training complete!")
