from angkorice_vision.exception.exception import AngkoriceVisionException
from angkorice_vision.logging.logger import logging
import numpy as np
import base64
import time
import json
import os
import sys
import cv2
import zipfile 
from io import BytesIO


def image_to_base64(img_rgb, ext=".jpg"):
    """
    Convert RGB image (numpy array) to base64 by using CV2 (pillow)
    """
    try:
        logging.info("---Start covert CV2 images to Base64---")
        # Convert RGB to BGR for OpenCV encoding
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        sucess, buffer = cv2.imencode(ext, img_bgr)
        if not sucess:
            raise ValueError(f"Failed to encode image")
        logging.info("---Completed to covert CV2 images to Base64---")
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        raise AngkoriceVisionException(e, sys)

def save_classified_images_to_json(results, images, collection_name="Rice Leaf Classification"):
    """
    A function to convert predicted results into JSON format.
    Args:
        results: from classifier.predict()
    Return:
        List of dict : [{RGB numpy array}, {pred_class}, {confidense}]
    """
    try:
        logging.info("---Start saving results to JSON format---")
        records = []

        for i, (result, img) in enumerate(zip(results, images)):
            encoded_img = image_to_base64(img)

            records.append({
                "name" : f"classified_image_{i+1}.jpg",
                "class" : result["label"],
                "confidence" : round(result["confidence"], 2),
                "content" : encoded_img,
                "timestamp" : int(time.time())
            })
        logging.info("---Completed converting predicted results into JSON format")
        return {
            "collection_name" : collection_name,
            "images" : records
        }

    except Exception as e:
        raise AngkoriceVisionException(e, sys)

def create_zipfile_of_images(images):
    try:
        logging.info("---Start downloading results in zip file---")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, img in enumerate(images):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Encode image to JPG memory
                _, buffer = cv2.imencode(".jpg", img_bgr)
                if not _ :
                    raise ValueError("Fialed to encode image.")
                # Write bytes to zip
                zipf.writestr(f"classified_image_{i+1}.jpg", buffer.tobytes())
        
        zip_buffer.seek(0)
        logging.info("---Completed creating zip file---")
        return zip_buffer
    except Exception as e:
        raise AngkoriceVisionException(e,sys)

# Notes 1:
# retval, buf = cv2.imencode(ext, img[, params])
## ext: A string representing the file extension that defines the output format (e.g., '.jpg', '.png', '.bmp').
## retval: A boolean flag that is True if the image was successfully encoded, and False otherwise.
## buf: The output buffer, a one-dimensional NumPy array of unsigned 8-bit integers (uint8) containing the encoded image as a byte array. 