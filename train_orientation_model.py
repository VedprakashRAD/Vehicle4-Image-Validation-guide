import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_model(num_classes, base_model_name="resnet50"):
    """Create a model for vehicle orientation classification"""
    if base_model_name.lower() == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def unfreeze_layers(model, num_layers=20):
    """Unfreeze the last num_layers of the base model for fine-tuning"""
    # Unfreeze the last num_layers layers
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    
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

def train_model(data_dir="processed_dataset", batch_size=32, epochs=30, fine_tune_epochs=15, base_model_name="resnet50"):
    """Train a vehicle orientation classification model"""
    print(f"Training vehicle orientation model using {base_model_name}...")
    
    # Create output directory for model and results
    output_dir = f"orientation_model_{base_model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,  # Don't flip horizontally as it changes orientation
        validation_split=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names and number of classes
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create model
    model = create_model(num_classes, base_model_name)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
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
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)
    
    # Initial training phase with frozen base model
    print("\nTraining with frozen base layers...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot initial training history
    plot_history(history, os.path.join(output_dir, 'initial_training_history.png'))
    
    # Fine-tuning phase
    print("\nFine-tuning the model...")
    model = unfreeze_layers(model)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with unfrozen layers
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot fine-tuning history
    plot_history(fine_tune_history, os.path.join(output_dir, 'fine_tuning_history.png'))
    
    # Save the final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Evaluate the model
    print("\nEvaluating the model...")
    validation_generator.reset()
    y_true = validation_generator.classes
    y_pred = np.argmax(model.predict(validation_generator), axis=1)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"\nTraining complete! Model and results saved to {output_dir}")

def main():
    # Train the model
    train_model(data_dir="processed_dataset", batch_size=16, epochs=30, fine_tune_epochs=15, base_model_name="resnet50")

if __name__ == "__main__":
    main() 