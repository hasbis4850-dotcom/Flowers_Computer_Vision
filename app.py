import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Flower Vision App", layout="wide")
st.title("🌸 Flower Recognition and Image Processing")
st.write("Upload a flower image, predict its class, and apply image-processing techniques.")

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "flowers_model.keras"
CLASS_PATH = BASE_DIR / "class_names.txt"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file flowers_model.keras nahi mili. Isay app.py ke same folder mein rakhein.")
    st.stop()

model = load_model()

if CLASS_PATH.exists():
    class_names = CLASS_PATH.read_text().splitlines()
else:
    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Upload image
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    st.subheader("1. Original Image")
    st.image(original, caption="Uploaded Image", use_container_width=True)

    # Flower prediction
    resized_for_model = cv2.resize(original, (128, 128))
    model_input = np.expand_dims(resized_for_model, axis=0)
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    st.subheader("2. Flower Prediction")
    st.success(f"Predicted Flower: {class_names[predicted_index]}")
    st.write(f"Model confidence: {probabilities[predicted_index] * 100:.2f}%")

    # Image processing options
    st.subheader("3. Image Processing")
    technique = st.selectbox(
        "Choose a processing technique",
        [
            "Resize",
            "Average Blur",
            "Gaussian Blur",
            "Grayscale",
            "Edge Detection"
        ]
    )

    if technique == "Resize":
        width = st.slider("Width", 100, 800, 300, 50)
        height = st.slider("Height", 100, 800, 300, 50)
        processed = cv2.resize(original, (width, height))

    elif technique == "Average Blur":
        processed = cv2.blur(original, (7, 7))

    elif technique == "Gaussian Blur":
        processed = cv2.GaussianBlur(original, (7, 7), 0)

    elif technique == "Grayscale":
        processed = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    else:
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed = cv2.Canny(gray, 100, 200)

    # Show original and processed images
    left, right = st.columns(2)

    with left:
        st.image(original, caption="Original Image", use_container_width=True)

    with right:
        st.image(processed, caption=technique, use_container_width=True)

    # Histogram
    st.subheader("4. Image Histogram")
    fig, ax = plt.subplots()

    if len(original.shape) == 3:
        for i, color in enumerate(["red", "green", "blue"]):
            histogram = cv2.calcHist(
                [original], [i], None, [256], [0, 256]
            )
            ax.plot(histogram, color=color, label=color.capitalize())
        ax.legend()
    else:
        ax.hist(original.ravel(), bins=256, range=(0, 256))

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.set_title("Original Image Histogram")
    st.pyplot(fig)
    plt.close(fig)


from io import BytesIO

buffer = BytesIO()
Image.fromarray(processed).save(buffer, format="PNG")

st.download_button(
    "Download Processed Image",
    data=buffer.getvalue(),
    file_name="processed_image.png",
    mime="image/png"
)
