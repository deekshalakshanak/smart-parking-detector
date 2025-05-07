import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import tempfile
import time

API_KEY = "2PK2ci7yrB4NXchB9FRW"
MODEL_ID = "parking-spaces-ezhxz/5"

# Dark theme setup
st.set_page_config(page_title="Parking Slot Detector", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    body { background-color: #0f1117; color: #fff; }
    .stApp { background-color: #0f1117; color: white; }
    .css-1d391kg { background-color: #0f1117; }
    .stButton > button { background-color: #262730; color: white; border-radius: 8px; }
    .stFileUploader { color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 Smart Parking Slot Detector")
st.write("Upload a parking lot image to detect **Occupied** and **Vacant** slots.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing image... Please wait"):
        # Save image to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image_path = tfile.name

        # Open image with OpenCV
        image = cv2.imread(image_path)

        with open(image_path, "rb") as image_file:
            response = requests.post(
                f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}&confidence=50",
                files={"file": image_file},
            )

        if response.status_code == 200:
            result = response.json()
            vacant_count = 0
            occupied_count = 0

            for prediction in result['predictions']:
                x = int(prediction['x'] - prediction['width'] / 2)
                y = int(prediction['y'] - prediction['height'] / 2)
                w = int(prediction['width'])
                h = int(prediction['height'])
                class_name = prediction['class']
                confidence = prediction['confidence']

                color = (0, 255, 0) if class_name.lower() == "empty" else (0, 0, 255)
                if class_name.lower() == "empty":
                    vacant_count += 1
                else:
                    occupied_count += 1

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{class_name} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert image to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
            st.image(image_rgb, caption="Detected Slots", use_container_width=True)


            st.success(f"✅ Detection Complete!")
            st.markdown(f"**Vacant Slots:** {vacant_count}")
            st.markdown(f"**Occupied Slots:** {occupied_count}")
        else:
            st.error(f"Error {response.status_code}: Could not analyze the image.")
