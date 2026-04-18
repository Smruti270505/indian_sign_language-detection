import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import cv2
import os
import av
import time
import threading
import queue
import pyttsx3
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# MUST BE FIRST
st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")

# =========================
# CONFIG
# =========================
IMG_SIZE   = 160
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "resnet50_model.pth")
DATA_DIR   = r"D:\majorprojectdataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CLASS LABELS
# =========================
classes     = sorted([d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))])
idx2word    = {i: cls for i, cls in enumerate(classes)}
NUM_CLASSES = len(classes)

# =========================
# TRANSFORM
# =========================
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =========================
# PREDICT
# =========================
def predict_image(pil_img):
    tensor = val_tfm(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out  = model(tensor)
        pred = out.argmax(1).item()
    return idx2word[pred]

# =========================
# THREADED TTS
# =========================
_speak_q    = queue.Queue(maxsize=1)
_last_spoke = 0.0

def _speaker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while True:
        text = _speak_q.get()
        engine.say(text)
        engine.runAndWait()
        _speak_q.task_done()

threading.Thread(target=_speaker, daemon=True).start()

def speak_async(text):
    global _last_spoke
    if not st.session_state.get("speak_on", True): return
    if time.time() - _last_spoke < 2.5: return
    _last_spoke = time.time()
    try: _speak_q.put_nowait(text)
    except queue.Full: pass

# =========================
# WEBCAM PROCESSOR
# =========================
class ISLProcessor(VideoProcessorBase):
    def __init__(self):
        self.label      = "—"
        self.last_label = ""

    def recv(self, frame):
        img     = frame.to_ndarray(format="bgr24")
        img     = cv2.flip(img, 1)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label      = predict_image(pil_img)
        self.label = label
        if label != self.last_label:
            speak_async(label)
            self.last_label = label
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 75), (0, 0, 0), -1)
        cv2.putText(img, label, (20, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5,
                    (74, 222, 128), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# UI
# =========================
st.title("🤟 Indian Sign Language Recognition")
st.caption(f"Model: ResNet50  |  Classes: {NUM_CLASSES}  |  Device: {DEVICE}")

if "history"  not in st.session_state: st.session_state.history  = []
if "speak_on" not in st.session_state: st.session_state.speak_on = True

tab_cam, tab_upload = st.tabs(["📷 Live Webcam", "🖼️ Upload Image"])

# ── WEBCAM TAB ──
with tab_cam:
    col_cam, col_info = st.columns([2, 1])
    with col_cam:
        ctx = webrtc_streamer(
            key="isl-webcam",
            video_processor_factory=ISLProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col_info:
        st.markdown("### Prediction")
        pred_ph = st.empty()
        hist_ph = st.empty()
        st.markdown("---")
        st.session_state.speak_on = st.toggle("🔊 Speak labels", value=True)
        if st.button("🗑️ Clear history"):
            st.session_state.history = []

    if ctx.video_processor:
        label = ctx.video_processor.label
        if label not in ("—", "", None) and (
            not st.session_state.history or
            st.session_state.history[-1] != label
        ):
            st.session_state.history.append(label)
            st.session_state.history = st.session_state.history[-30:]
        pred_ph.markdown(
            f"""<div style='background:#111827;border:1px solid #1e2530;border-radius:12px;
                padding:24px;text-align:center;'>
                <div style='font-size:2.5rem;font-weight:700;color:#60a5fa;'>{label}</div>
                </div>""",
            unsafe_allow_html=True)
        if st.session_state.history:
            pills = "".join(
                f'<span style="display:inline-block;background:#1e2530;color:#e8eaed;'
                f'border-radius:20px;padding:4px 14px;margin:3px;font-size:0.95rem;'
                f'font-weight:600;">{h}</span>'
                for h in reversed(st.session_state.history[-10:])
            )
            hist_ph.markdown(f"**Recent:**<br>{pills}", unsafe_allow_html=True)
        time.sleep(0.05)
        st.rerun()

# ── UPLOAD TAB ──
with tab_upload:
    uploaded = st.file_uploader("Upload an ISL image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Predicting..."):
                label = predict_image(image)
            st.success(f"Prediction: **{label}**")
