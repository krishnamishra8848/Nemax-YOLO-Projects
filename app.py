import av
import time
import threading
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import WebRtcMode, webrtc_streamer

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs_cls" / "employee_cls_10ep" / "weights" / "best.pt"

st.set_page_config(page_title="Employee Classifier", layout="centered")
st.title("Employee Activity Classifier")
st.caption("Realtime webcam prediction + image upload test.")


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
tts_lock = threading.Lock()
tts_engine = None
if pyttsx3 is not None:
    try:
        tts_engine = pyttsx3.init()
    except Exception:
        tts_engine = None
if "last_spoken_label" not in st.session_state:
    st.session_state.last_spoken_label = ""
if "last_spoken_time" not in st.session_state:
    st.session_state.last_spoken_time = 0.0

st.subheader("Realtime Webcam")
st.write("Click 'START' below to allow camera and run live prediction.")
if tts_engine is None:
    st.info("Voice output unavailable in this environment. Prediction will still run.")
min_conf = st.slider("Voice confidence threshold", 0.0, 1.0, 0.40, 0.05)
cooldown_sec = st.slider("Voice cooldown (sec)", 0.0, 5.0, 0.5, 0.1)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = model(img, verbose=False)[0]

    if result.probs is not None:
        top1_idx = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        label = result.names[top1_idx]

        now = time.time()
        should_speak = (
            top1_conf >= min_conf
            and (now - st.session_state.last_spoken_time) > cooldown_sec
        )
        if should_speak:
            st.session_state.last_spoken_label = label
            st.session_state.last_spoken_time = now

            def speak_prediction(text):
                if tts_engine is None:
                    return
                with tts_lock:
                    tts_engine.say(text)
                    tts_engine.runAndWait()

            threading.Thread(target=speak_prediction, args=(label,), daemon=True).start()

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="employee-realtime",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

st.divider()
st.subheader("Upload Image Test")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict Image", type="primary"):
        result = model(image, verbose=False)[0]
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        top1_label = result.names[top1_idx]

        st.success(f"Prediction: {top1_label}")
        st.write(f"Confidence: {top1_conf:.2%}")
        if tts_engine is not None:
            with tts_lock:
                tts_engine.say(top1_label)
                tts_engine.runAndWait()
