import streamlit as st
import numpy as np
import cv2
import sys
import os
import json
import matplotlib.pyplot as plt
import time
from angkorice_vision.logging.logger import logging
from angkorice_vision.exception.exception import AngkoriceVisionException
from angkorice_vision.prediction.prediction import ImageClassifier
from angkorice_vision.utils.utils import image_to_base64, save_classified_images_to_json, create_zipfile_of_images
st.markdown("""
<style>
/* Make columns full-width with padding for grid effect */
div[data-testid="column"] {
    padding: 8px !important;
}

/* Style all images inside columns */
div[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    width: 100% !important;
    height: auto !important;
    object-fit: cover;
}

/* Hover effect */
div[data-testid="stImage"] img:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}

/* Optional: caption styling */
div[data-testid="stImage"] > div:nth-child(2) {
    font-size: 13px;
    color: #333;
    text-align: center;
    padding: 4px 0;
}
</style>
""", unsafe_allow_html=True)


cwd = os.getcwd()
#model_path = os.path.join(cwd, "model", "xception_fine_tuned_rice_leaf_model_30.keras")
model_path = r"xception_fine_tuned_rice_leaf_model_30.keras"
class_names = {
    0: 'Bacterialblight', 
    1: 'Blast', 
    2: 'Brownspot', 
    3: 'Healthyleaf', 
    4: 'Leafscald', 
    5: 'Sheathblight', 
    6: 'Tungro'
}

classifier = ImageClassifier(model_path,class_names)

st.set_page_config(page_title ="🌾 Angkorice Vision", layout="wide")

st.markdown("""
<div style="text-align:center; margin-top:10px;">
    <h1 style="font-size:48px; margin-bottom:5px;">🌾 Angkorice Vision</h1>
</div>
""", unsafe_allow_html=True)


st.subheader("Rice Leaf Disease Detection and Advisory System", text_alignment="center")

def init_session_state():
    session_defaults = {
        "uploader_key": 0,
        "results" : None,
        "classified_images" : None,
        "prediction_done" : False,
        "download_options": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_key():
    st.session_state.uploader_key += 1
    st.session_state.results = None
    st.session_state.classified_images = None
    st.session_state.prediction_done = False

init_session_state()


# Upload Botton
uploaded_images = st.file_uploader("Upload your rice leaf images 🍚🍃 :", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                    accept_multiple_files=True, key=f"Uploader_{st.session_state.uploader_key}") # key changed--> uploader reset

if uploaded_images:
    try:
        with st.spinner("Loading image"):
            cols = st.columns(3)
            if uploaded_images:
                # Clear Uploaded Botton
                st.button("Clear Uploaded Images", on_click=update_key, icon ="🔁")
            # Convert raw image to byte file
            for idx, uploaded_image in enumerate(uploaded_images):
                file_bytes = np.asanyarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                # Decode byte file image and covert BGR to RBG
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is None:
                    st.error(f"Failed to upload {uploaded_image.name}")
                    continue
                # Define max width on UI
                max_width = 500
                h, w = img.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    scaled_size = (max_width, int(h * scale))
                    resized_img_rgb = cv2.resize(img, scaled_size)
                else:
                    resized_img_rgb = img
    
                with cols[idx % 3]:
                    st.image(resized_img_rgb, caption=f"Uploaded Image_{idx+1}", width="content")
         
        st.success("Sucessfuly uploaded image!")       
    except Exception as e:
        st.error("Error occured during image")
        raise AngkoriceVisionException(e, sys)
    
    if st.button("Predict", type="primary"):
        try:
            with st.spinner("Inferencing the images...⏳"):
                grid_cols = st.columns(3)
                image_batch, raw_images = classifier.preprocess_uploaded_images(uploaded_images)
                results = classifier.predict(uploaded_images)
                classified_images = []
                
                for i, result in enumerate(results):
                    pred_label = result["label"]
                    confidence = result["confidence"]
                    classified_img = raw_images[i].copy()
                    h, w = classified_img.shape[:2]
                    font_scale = max(0.50, w/600)
                    thichness = max(1, round(w // 550))
                    cv2.putText(
                        classified_img, 
                        f"{pred_label} : Score: {confidence:.2f}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 0, 0),
                        thichness,
                        cv2.LINE_AA)
                    # Append labeled images to classified images list
                    classified_images.append(classified_img)
                    with grid_cols[i % 3]:
                        st.image(classified_img,caption=f"Classifed Image_{i+1}", width="content")
                # Update initial_session_state --> Persist across reruns (UI's simply behavors)
                st.session_state.results = results
                st.session_state.classified_images = classified_images
                st.session_state.prediction_done = True
                
                                      
        except Exception as e:
            st.error("Error occured during image inferencing.")
            raise AngkoriceVisionException(e,sys)
if st.session_state.prediction_done:
                with st.expander("Export Options :", expanded=True, icon="📤"):
                    
                    download_options = st.radio("Choose an option :", ("Save Results in JSON", "Save Results in Zip File"), horizontal=True, key="download_options")
                    if st.session_state.download_options == "Save Results in JSON":
                        json_data = save_classified_images_to_json(
                            st.session_state.results,
                            st.session_state.classified_images
                        )
                        json_str = json.dumps(json_data, indent=2)
                        st.download_button(
                            label="💾 Download JSON Results",
                            data=json_str,
                            file_name="rice_leaf_predictions.json",
                            mime="application/json"
                        )
                    else:
                        zip_file = create_zipfile_of_images(
                            st.session_state.classified_images
                        )
                        st.download_button(
                            label="⬇️ Download Classified Images",
                            data=zip_file,
                            file_name="rice_leaf_classified_images.zip",
                            mime="application/zip"
                        )

st.markdown(
    """
    **សេចក្តីប្រកាសបដិសេធ៖** កម្មវិធីវេបនេះត្រូវបានបង្កើតឡើងសម្រាប់គោលបំណងអប់រំប៉ុណ្ណោះ។ សម្រាប់ការវាយតម្លៃជំងឺលើស្លឹកស្រូវ សូមពិគ្រោះជាមួយអ្នកជំនាញកសិកម្ម ឬអង្គភាព/បុគ្គលដែលមានការអនុញ្ញាតជាផ្លូវការ។  

    **Disclaimer:** This web application is intended for educational purposes only and should not be used for real-world diagnosis. For accurate rice leaf disease evaluation, please consult qualified agricultural experts or authorized institutions.
    """
)
st.write("💻 Developed by: Chea Vearo")
            


