import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import sys
import cv2
from angkorice_vision.exception.exception import AngkoriceVisionException
from angkorice_vision.logging.logger import logging


class ImageClassifier:
        def __init__(self, model_path, class_names=None):

            try:
                logging.info("Initiate the model loading.")
                self.model = load_model(model_path, compile=False)
                self.class_names = class_names

                if class_names is None:
                    class_names = {
                        0: 'Bacterialblight', 
                        1: 'Blast', 
                        2: 'Brownspot', 
                        3: 'Healthyleaf', 
                        4: 'Leafscald', 
                        5: 'Sheathblight', 
                        6: 'Tungro'
                    }
                else:
                    self.class_names = class_names
            except Exception as e:
                raise AngkoriceVisionException(e, sys)
            
        def predict(self, uploaded_images):
            """
            A function to inference the images.
            Args:
                images : Uploaded images for predictions.
            Return:
                List of dict :  [{label, confidence}]
            """
            try:
                logging.info("Inferencing the batch of uploaded images.")
                images,raw_images = self.preprocess_uploaded_images(uploaded_images)
                # Batch prediction (No flatten & no softmax, they are already included in model structure.)
                predictons = self.model.predict_on_batch(images) # Shape (batch_size, nun_classes)
                # Get the predicted class index and confidence score
                predicted_class_idx = tf.argmax(predictons, axis=1).numpy()
                confidence_scores = tf.reduce_max(predictons, axis=1).numpy()
                # Example: predictions = tf.constant([0.2, 0.3, 0.5], [0.1, 0.8, 0.1])
                # tf.argmax(predictions, axis=1).numpy()--> [2, 1] | Which class?
                # tf.reduce_max(prediction, axis=1).numpy()--> [0.5, 0.8] | Extract the max value itself

                results = []

                # Predicted classes
                for i in range(len(images)):
                    results.append({
                        "label": self.class_names[int(predicted_class_idx[i])],
                        "confidence": float(confidence_scores[i] * 100)
                    })

                return results
            except Exception as e:
                raise AngkoriceVisionException(e,sys)
        
        def preprocess_uploaded_images(self, uploaded_images, img_size=(299, 299)):
            """
            Preprocess uploaded images for inference.

            Args:
                uploaded_images: list of Streamlit uploaded_file objects
                img_size: tuple, target size (height, width)

            Returns:
                images: np.array of shape (batch, img_size[0], img_size[1], 3)
                raw_images: list of original RGB images (for visualization)
            """
            try:
                images = []
                raw_images = []

                for uploaded_image in uploaded_images:
                    uploaded_image.seek(0)  # reset file pointer
                    # Read file bytes safely
                    file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
                    if file_bytes.size == 0:
                        continue  # skip empty files

                    # Decode image
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        continue  # skip unreadable files

                    # Convert BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Save raw image for display
                    raw_images.append(img.copy())

                    # Resize for model input
                    img_resized = cv2.resize(img, img_size)
                    images.append(img_resized.astype(np.float32))

                if len(images) == 0:
                    # Warn user instead of crashing
                    return np.array([]), []

                return np.stack(images), raw_images

            except Exception as e:
                raise AngkoriceVisionException(e, sys)

