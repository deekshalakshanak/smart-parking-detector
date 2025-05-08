import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import tempfile

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

        try:
            # Load image using PIL
            pil_image = Image.open(uploaded_file).convert("RGB")
            image = np.array(pil_image)
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")
            st.stop()

        # Convert to BGR for OpenCV use
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tfile:
            cv2.imwrite(tfile.name, image_bgr)

            with open(tfile.name, "rb") as image_file:
                response = requests.post(
                    f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}&confidence=50",
                    files={"file": image_file},
                )

        if response.status_code == 200:
            result = response.json()
            vacant_count = 0
            occupied_count = 0

            for pred in result.get("predictions", []):
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                w = int(pred["width"])
                h = int(pred["height"])
                class_name = pred["class"]
                confidence = pred["confidence"]

                color = (0, 255, 0) if class_name.lower() == "empty" else (0, 0, 255)
                if class_name.lower() == "empty":
                    vacant_count += 1
                else:
                    occupied_count += 1

                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image_bgr, f"{class_name} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Final conversion to RGB
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                st.write(f"✅ Image successfully processed. Shape: {image_rgb.shape}")
                st.image(image_rgb, caption="Detected Slots", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Failed to render image: {e}")
                st.stop()

            st.success("✅ Detection Complete!")
            st.markdown(f"**Vacant Slots:** {vacant_count}")
            st.markdown(f"**Occupied Slots:** {occupied_count}")
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
