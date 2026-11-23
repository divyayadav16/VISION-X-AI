
import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
from io import BytesIO


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="VISION-X | AI Intelligence Lab",
    page_icon="🔷",
    layout="wide"
)


# -------------------- CUSTOM DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d18, #101a2b);
    color: #f1f5f9;
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] {
    background: #0a1220;
    border-right: 1px solid #26364d;
}
div[data-testid="stMetric"] {
    background: #111f32;
    border: 1px solid #263b55;
    padding: 17px;
    border-radius: 13px;
}
div[data-testid="stMetricValue"] {
    color: #27d9ff;
}
.stButton > button {
    background: linear-gradient(90deg, #00c6e8, #477bff);
    color: #06111e;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}
.stButton > button:hover {
    color: white;
    border: 1px solid #27d9ff;
}
.vx-card {
    background: #111f32;
    border: 1px solid #263b55;
    border-radius: 14px;
    padding: 20px;
}
.vx-label {
    color: #27d9ff;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}
.vx-muted {
    color: #98a9be;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL LOADER --------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🔷 VISION-X")
    st.caption("COMPUTER VISION INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Detection Studio", "About Project"]
    )

    st.divider()
    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05
    )

    st.markdown("---")
    st.markdown("🟢 **Application Ready**")
    st.caption("Model: YOLO11 Nano")
    st.caption("Build: V1.0")


# -------------------- HEADER --------------------

st.markdown(
    '<p class="vx-label">AI COMPUTER VISION PLATFORM</p>',
    unsafe_allow_html=True
)

st.title("Vision Intelligence Dashboard")

st.markdown(
    '<p class="vx-muted">Detect. Analyze. Understand visual data.</p>',
    unsafe_allow_html=True
)

st.divider()


# -------------------- ABOUT PAGE --------------------

if page == "About Project":
    st.subheader("About VISION-X")

    st.write("""
    VISION-X is an AI-powered computer vision application.

    It uses a pretrained YOLO11 Nano object detection model
    to identify and locate objects in uploaded images.

    **Technology stack**
    - Python
    - Streamlit
    - Ultralytics YOLO11
    - OpenCV
    - Pandas
    - Plotly
    - Pillow

    **Core capabilities**
    - Image upload and preview
    - Object detection
    - Confidence scoring
    - Object frequency analysis
    - Annotated image export
    """)

    st.stop()


# -------------------- OVERVIEW --------------------

st.markdown("### ◈ Detection Workspace")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Choose an image containing objects you want to detect."
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("#### Uploaded Image")
    st.image(image, use_container_width=True)

    run_detection = st.button(
        "🚀 Run AI Object Detection",
        use_container_width=True
    )

    if run_detection:
        try:
            with st.spinner(
                "Loading YOLO model and analyzing your image..."
            ):
                model = load_model()

                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                # Convert annotated image from BGR to RGB
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_BGR2RGB
                )

                # Extract detection information
                records = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())
                        label = result.names[class_id]

                        records.append({
                            "Object": label,
                            "Confidence": round(score * 100, 2)
                        })

                df = pd.DataFrame(
                    records,
                    columns=["Object", "Confidence"]
                )

            st.session_state["annotated_image"] = annotated_rgb
            st.session_state["detections"] = df

            st.success("AI analysis completed successfully!")

        except Exception as error:
            st.error(f"Detection failed: {error}")
            st.info(
                "Check your internet connection and package installation. "
                "The pretrained model may need to download on first use."
            )


# -------------------- RESULTS --------------------

if "detections" in st.session_state:

    df = st.session_state["detections"]
    annotated = st.session_state["annotated_image"]

    st.divider()
    st.markdown("### ◈ AI Detection Results")

    total_objects = len(df)
    unique_objects = df["Object"].nunique()

    average_confidence = (
        df["Confidence"].mean() if total_objects else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Objects Detected", total_objects)
    c2.metric("Unique Classes", unique_objects)
    c3.metric("Average Confidence", f"{average_confidence:.1f}%")

    st.markdown("#### Annotated Detection Image")

    st.image(
        annotated,
        use_container_width=True,
        caption="YOLO11 object detection output"
    )

    image_buffer = BytesIO()
    Image.fromarray(annotated).save(image_buffer, format="PNG")

    st.download_button(
        "⬇️ Download Annotated Image",
        data=image_buffer.getvalue(),
        file_name="vision_x_detection.png",
        mime="image/png",
        use_container_width=True
    )

    st.markdown("#### Detection Details")

    if total_objects > 0:
        display_df = df.copy()
        display_df["Confidence"] = (
            display_df["Confidence"].astype(str) + "%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Object Frequency Analytics")

        counts = (
            df["Object"]
            .value_counts()
            .rename_axis("Object")
            .reset_index(name="Count")
        )

        chart = px.bar(
            counts,
            x="Object",
            y="Count",
            color="Count",
            template="plotly_dark",
            title="Detected Objects by Category"
        )

        chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(chart, use_container_width=True)

    else:
        st.warning(
            "No objects were detected above the selected confidence threshold."
        )

else:
    st.info(
        "Upload an image and click Run AI Object Detection "
        "to begin your analysis."
    )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "VISION-X | AI Computer Vision Intelligence Lab | Built with Python"
)
