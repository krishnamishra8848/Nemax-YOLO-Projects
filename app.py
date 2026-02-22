from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs_cls" / "employee_cls_10ep" / "weights" / "best.pt"

st.set_page_config(page_title="Employee Classifier", layout="centered")
st.title("Employee Activity Classifier")
st.caption("Capture from camera or upload image for instant prediction.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Camera Capture")
st.write("Tap camera, take photo, and prediction runs automatically.")
captured = st.camera_input("Open camera")

if captured is not None:
    cam_image = Image.open(captured).convert("RGB")
    st.image(cam_image, caption="Captured image", use_container_width=True)

    cam_result = model(cam_image, verbose=False)[0]
    cam_probs = cam_result.probs
    cam_top1_idx = int(cam_probs.top1)
    cam_top1_conf = float(cam_probs.top1conf)
    cam_top1_label = cam_result.names[cam_top1_idx]

    st.success(f"Prediction: {cam_top1_label}")
    st.write(f"Confidence: {cam_top1_conf:.2%}")

st.divider()
st.subheader("Upload Image Test")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    result = model(image, verbose=False)[0]
    probs = result.probs
    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)
    top1_label = result.names[top1_idx]

    st.success(f"Prediction: {top1_label}")
    st.write(f"Confidence: {top1_conf:.2%}")
