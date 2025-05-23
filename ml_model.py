import numpy as np
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

class VehicleSideClassifier:
    """
    A class to classify which side of a vehicle is shown in an image.
    Uses a pre-trained MobileNetV2 model fine-tuned for vehicle side classification.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier
        
        Args:
            model_path: Optional path to a saved model. If None, a new model will be created.
        """
        self.sides = ["front", "rear", "left", "right"]
        self.model = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading vehicle side classifier from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Creating new vehicle side classifier model")
            self.model = self._create_model()
            
        print("Vehicle side classifier initialized")
    
    def _create_model(self):
        """Create a new model for vehicle side classification"""
        # Use a pre-trained model as the base
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),  # Increased from 128 to 256
            tf.keras.layers.Dropout(0.3),  # Increased dropout for better generalization
            tf.keras.layers.Dense(128, activation='relu'),  # Added another layer
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: front, rear, left, right
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, img):
        """
        Preprocess an image for the model
        
        Args:
            img: PIL Image object
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Resize to the input size expected by the model
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.shape[2] == 1:
            img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
        
        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, img):
        """
        Predict which side of the vehicle is shown in the image
        
        Args:
            img: PIL Image object
            
        Returns:
            str: The predicted side ("front", "rear", "left", "right")
            float: Confidence score (0-1)
        """
        # For an untrained model, use improved heuristic matching instead of random
        if not hasattr(self, 'model') or self.model is None:
            from image_utils import check_vehicle_perspective
            
            # Check each side and get confidence scores
            scores = {}
            for side in self.sides:
                is_correct, confidence, _ = check_vehicle_perspective(img, side)
                scores[side] = confidence * (1.5 if is_correct else 0.5)
            
            # Get the side with highest score
            predicted_side = max(scores, key=scores.get)
            confidence = scores[predicted_side]
            
            return predicted_side, confidence
        
        # 1. Preprocess the image
        img_array = self.preprocess_image(img)
        
        # 2. Make prediction with the model
        predictions = self.model.predict(img_array, verbose=0)
        
        # 3. Get the class with highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # 4. Return the corresponding side and confidence
        return self.sides[predicted_class], float(confidence)
    
    def train(self, train_data, validation_data, epochs=10, save_path=None):
        """
        Train the model with vehicle side images
        
        Args:
            train_data: TensorFlow dataset for training
            validation_data: TensorFlow dataset for validation
            epochs: Number of training epochs
            save_path: Path to save the trained model
            
        Returns:
            History object from model.fit()
        """
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._create_model()
            
        # Define callbacks
        callbacks = []
        
        if save_path:
            checkpoint_path = os.path.join(save_path, 'best_model.h5')
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, 
                monitor='val_accuracy',
                verbose=1, 
                save_best_only=True,
                mode='max'
            )
            callbacks.append(checkpoint)
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Train the model
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        # Save the final model if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.model.save(os.path.join(save_path, 'final_model.h5'))
            
        return history


def prepare_dataset(image_dir, batch_size=32):
    """
    Prepare a TensorFlow dataset from directory structure
    
    Args:
        image_dir: Path to the directory containing subdirectories for each class
        batch_size: Batch size for training
        
    Returns:
        TensorFlow dataset
    """
    # Define preprocessing and augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    # Create dataset
    return tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=["front", "rear", "left", "right"],
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
        interpolation="bilinear",
        crop_to_aspect_ratio=True,
    ).map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE) 