import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training.log'
)

# Set memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
except:
    print("No GPU found or error setting memory growth")

def create_model(num_classes, model_name="efficientnet"):
    """Create a model for vehicle orientation classification"""
    input_shape = (224, 224, 3)
    
    if model_name.lower() == "efficientnet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.lower() == "resnet":
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {model_name}")
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def unfreeze_model(model, base_model, num_layers=30):
    """Unfreeze the last num_layers of the base model for fine-tuning"""
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all the layers except the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    return model

def plot_history(history, save_path="training_history.png"):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(data_dir="processed_large_dataset", 
                batch_size=32, 
                initial_epochs=30, 
                fine_tune_epochs=20, 
                model_name="efficientnet",
                img_size=224,
                validation_split=0.2):
    """Train a vehicle orientation classification model"""
    print(f"Training vehicle orientation model using {model_name}...")
    
    # Create timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create output directory for model and results
    output_dir = f"orientation_model_{model_name}_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create TensorBoard log directory
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Data augmentation for training with error handling
    try:
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,  # Don't flip horizontally as it changes orientation
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Load training data with error handling
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            interpolation='bilinear'  # Use bilinear interpolation for better results
        )
        
        # Load validation data with error handling
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            interpolation='bilinear'  # Use bilinear interpolation for better results
        )
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
    
    # Get class names and number of classes
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training on {train_generator.samples} samples")
    print(f"Validating on {validation_generator.samples} samples")
    
    # Save class names for later use
    with open(os.path.join(output_dir, 'class_names.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Create model
    model, base_model = create_model(num_classes, model_name)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)
    
    # Initial training phase with frozen base model
    print("\nPhase 1: Training with frozen base layers...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=initial_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Phase 1 training completed in {training_time:.2f} seconds")
    
    # Plot initial training history
    plot_history(history, os.path.join(output_dir, 'initial_training_history.png'))
    
    # Save the initial model
    model.save(os.path.join(output_dir, 'initial_model.h5'))
    
    # Fine-tuning phase
    print("\nPhase 2: Fine-tuning the model...")
    model = unfreeze_model(model, base_model)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with unfrozen layers
    start_time = time.time()
    
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    fine_tuning_time = time.time() - start_time
    print(f"Phase 2 fine-tuning completed in {fine_tuning_time:.2f} seconds")
    
    # Plot fine-tuning history
    plot_history(fine_tune_history, os.path.join(output_dir, 'fine_tuning_history.png'))
    
    # Save the final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Evaluate the model
    print("\nEvaluating the model...")
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Get predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save training info
    with open(os.path.join(output_dir, 'training_info.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Initial epochs: {initial_epochs}\n")
        f.write(f"Fine-tuning epochs: {fine_tune_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Image size: {img_size}x{img_size}\n")
        f.write(f"Training samples: {train_generator.samples}\n")
        f.write(f"Validation samples: {validation_generator.samples}\n")
        f.write(f"Initial training time: {training_time:.2f} seconds\n")
        f.write(f"Fine-tuning time: {fine_tuning_time:.2f} seconds\n")
        f.write(f"Total training time: {training_time + fine_tuning_time:.2f} seconds\n")
    
    print(f"\nTraining complete! Model and results saved to {output_dir}")
    return output_dir

def main():
    # Train the model with improved parameters
    output_dir = train_model(
        data_dir="processed_large_dataset",
        batch_size=16,
        initial_epochs=30,
        fine_tune_epochs=20,
        model_name="efficientnet",
        img_size=224,
        validation_split=0.2
    )
    
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main() 