import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os, json, time, pandas as pd

# ──────────────────────────────────────────────
# CONFIG + THEME
# ──────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognition AI", page_icon="🤖", layout="centered")
st.markdown("""
<style>
body, .stApp { background-color:#0e1117; color:#fafafa; }
h1,h2,h3,h4,h5,h6 { color:#fafafa; }
.stButton>button{background-color:#262730;color:#fafafa;border-radius:8px;}
.stButton>button:hover{background-color:#3c4048;color:#fff;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FILES & DATA
# ──────────────────────────────────────────────
MODEL_PATH = "model/digit_recognition_model.keras"
FEEDBACK_DIR = "feedback_data"
HISTORY_FILE = "prediction_history.json"

os.makedirs(FEEDBACK_DIR, exist_ok=True)

def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

model = load_model()
history_data = load_history()

# ──────────────────────────────────────────────
# PAGE NAVIGATION
# ──────────────────────────────────────────────
page = st.sidebar.selectbox("📚 Select Page", ["Digit Recognition", "History"])

# ──────────────────────────────────────────────
# DIGIT RECOGNITION PAGE
# ──────────────────────────────────────────────
if page == "Digit Recognition":
    st.markdown("""
    <div style='text-align:center;'>
        <h1>🤖 Digit Recognition AI</h1>
        <p style='color:#9ca3af;'>Draw or upload a handwritten digit (0–9).</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload / Draw Section
    uploaded_file = st.file_uploader("📤 Upload an image", type=["png","jpg","jpeg"])
    st.markdown("**Or draw below:**")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=12,
        stroke_color="#ffffff",
        background_color="#000000",
        height=260,
        width=260,
        drawing_mode="freedraw",
        key="canvas"
    )
    predict_btn = st.button("🔍 Recognize Digit")

    # Helper functions
    def pil_from_canvas(canvas_obj):
        if canvas_obj is None or canvas_obj.image_data is None:
            return None
        img = Image.fromarray(canvas_obj.image_data.astype('uint8'), 'RGBA').convert('L')
        bbox = img.getbbox()
        if bbox: img = img.crop(bbox)
        return img

    def preprocess_pil(pil_img):
        img = pil_img.convert('L')
        img = ImageOps.invert(img)
        img.thumbnail((20, 20), Image.LANCZOS)
        new_img = Image.new('L', (28, 28), color=0)
        new_img.paste(img, ((28 - img.width)//2, (28 - img.height)//2))
        arr = np.array(new_img).astype('float32') / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr, new_img

    # Prediction
    if predict_btn:
        pil_img = Image.open(uploaded_file).convert('L') if uploaded_file else pil_from_canvas(canvas_result)
        if pil_img is None:
            st.warning("Please draw or upload a digit.")
        elif model:
            arr, new_img = preprocess_pil(pil_img)
            preds = model.predict(arr)
            digit = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0])) * 100.0

            st.image(new_img.resize((140,140)), caption="Processed Image", width=140)
            st.success(f"🎯 Predicted Digit: **{digit}**")
            st.info(f"Confidence: **{conf:.2f}%**")

            # Save history record
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted": digit,
                "confidence": round(conf, 2),
                "correct_label": None,
                "status": "Pending"
            }
            history_data.append(record)
            save_history(history_data)

            # Feedback / Correction
            correct_digit = st.text_input("If incorrect, enter correct digit (0–9):")
            if st.button("💾 Submit Correction"):
                if correct_digit.isdigit() and 0 <= int(correct_digit) <= 9:
                    img_path = os.path.join(FEEDBACK_DIR, f"{correct_digit}_{int(time.time())}.png")
                    new_img.save(img_path)

                    last_record = history_data[-1]
                    last_record["correct_label"] = int(correct_digit)
                    last_record["status"] = "Correct" if int(correct_digit)==last_record["predicted"] else "Incorrect"
                    save_history(history_data)

                    st.success(f"✅ Feedback saved (Label: {correct_digit})")
                else:
                    st.warning("Please enter a valid digit (0–9).")

# ──────────────────────────────────────────────
# HISTORY PAGE
# ──────────────────────────────────────────────
if page == "History":
    st.markdown("## 📊 Prediction History")
    if not history_data:
        st.info("No predictions yet.")
    else:
        df = pd.DataFrame(history_data)
        def color_row(row):
            color = "#14532d" if row["status"] == "Correct" else "#7f1d1d" if row["status"] == "Incorrect" else "#1e293b"
            return [f"background-color:{color};color:white;"]*len(row)
        st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<hr style='border:1px solid #3c4048;'>
<div style='text-align:center;color:#6b7280;'>
Made with ❤️ by <b>Abu Sufyan – Student</b> | Organization: <b>Abu Zar</b>
</div>
""", unsafe_allow_html=True)
