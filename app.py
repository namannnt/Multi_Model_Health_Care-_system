import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import os
import importlib.util
import time
import plotly.graph_objects as go
from typing import Tuple

# ----------------------------------------
# USER-CONFIGURABLE PATHS
# ----------------------------------------
# ----------------------------------------
# USER-CONFIGURABLE PATHS (Streamlit-safe)
# ----------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_DIR, "all three saved models")

RETINA_WEIGHTS = os.path.join(BASE_PATH, "best_model.pth")
PNEUMONIA_WEIGHTS = os.path.join(BASE_PATH, "densenet121_chest.pth")
ECG_WEIGHTS = os.path.join(BASE_PATH, "ecg_best_model_v4.pth")

# If your ECG model definition file exists in the repo:
ECG_MODEL_PY = os.path.join(BASE_DIR, "src", "03_train_model.py")

# ----------------------------------------
# CLASS LABELS
# ----------------------------------------
RETINA_CLASSES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]
PNEUMONIA_CLASSES = ["Normal", "Pneumonia"]
ECG_CLASSES = ["Other", "Myocardial Infarction (MI)"]

# ----------------------------------------
# UTILS / MODELS
# ----------------------------------------
@st.cache_resource
def load_retina_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(RETINA_CLASSES))
    model.load_state_dict(torch.load(RETINA_WEIGHTS, map_location=device))
    model.to(device).eval()
    return model

@st.cache_resource
def load_pneumonia_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, len(PNEUMONIA_CLASSES))
    model.load_state_dict(torch.load(PNEUMONIA_WEIGHTS, map_location=device))
    model.to(device).eval()
    return model

@st.cache_resource
def load_ecg_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ECG_MODEL_PY):
        raise FileNotFoundError(f"ECG model file not found: {ECG_MODEL_PY}")
    spec = importlib.util.spec_from_file_location("train_model", ECG_MODEL_PY)
    train_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_model)
    ECGNet = getattr(train_model, "ECGAttentionNet")
    model = ECGNet(num_classes=len(ECG_CLASSES)).to(device)
    model.load_state_dict(torch.load(ECG_WEIGHTS, map_location=device))
    model.eval()
    return model

def preprocess_image(img):
    # Convert uploaded file to bytes if needed
    if hasattr(img, "read"):
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif isinstance(img, Image.Image):
        # Convert PIL Image → NumPy array
        img = np.array(img.convert("RGB"))
    else:
        raise ValueError("Unsupported image format")

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert to tensor and normalize
    tensor = torch.from_numpy(img.transpose((2, 0, 1)))  # HWC → CHW
    tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])(tensor)
    return tensor.unsqueeze(0)
def generate_gradcam_image(model, image_tensor, target_layer):
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else np.zeros_like(heatmap)

    handle_fwd.remove()
    handle_bwd.remove()
    return heatmap, pred_class

def preprocess_ecg_from_array(arr: np.ndarray, target_length=1000):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr.reshape(-1, 1), (1, 12))
    elif arr.shape[0] == 12:
        arr = arr.T
    elif arr.shape[1] != 12:
        if arr.shape[0] > arr.shape[1]:
            arr = np.tile(arr, (1, 12))[:, :12]
        else:
            arr = np.tile(arr.T, (1, 12))[:, :12]

    resized = [cv2.resize(arr[:, ch].reshape(1, -1), (target_length, 1),
               interpolation=cv2.INTER_LINEAR).reshape(-1) for ch in range(12)]
    arr = np.stack(resized, axis=0)

    mean, std = arr.mean(axis=1, keepdims=True), arr.std(axis=1, keepdims=True) + 1e-8
    arr = (arr - mean) / std
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return tensor

def generate_ecg_saliency(model, input_tensor, device):
    model.zero_grad()
    x = input_tensor.clone().detach().to(device)
    x.requires_grad_(True)
    output = model(x)
    pred = torch.argmax(output, dim=1).item()
    score = torch.softmax(output, dim=1)[0, pred]
    score.backward()
    grads = x.grad.detach().cpu().numpy()[0]
    importance = np.mean(np.abs(grads), axis=0)
    importance = importance / (np.max(importance) + 1e-8)
    return importance, pred

# ----------------------------------------
# STYLES & HEADER
# ----------------------------------------
st.set_page_config(page_title="AI Health Analyzer", layout="wide")
st.markdown("""
    <style>
        .title {
            font-size:38px;
            font-weight:800;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: -10px;
        }
        .sub {
            color: #444444;
            font-size:14px;
            margin-top: 0px;
        }
        .card {
            padding:12px;
            border-radius:8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            background: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar / Navigation
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "🫁 Pneumonia", "⚡ ECG", "👁 Retina", "ℹ️ About"])

# Home
if page == "🏠 Home":
    st.markdown('<h1 class="title">🧬 AI Health Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Multi-model medical AI dashboard — Retina, Pneumonia & ECG (MI). Decision-support only.</div>', unsafe_allow_html=True)
    st.write("---")
    st.write("**Choose a page from the sidebar** to run models, upload images/ECGs and evaluate vitals.")
    st.write("Features included:")
    st.write("- Grad-CAM explainability for image models")
    st.write("- WHO/AHA-based vitals emergency checks")
    st.write("- Animated ECG playback & saliency overlay")
    st.write("- Patient summary cards & visual gauges")

# ---------------- PNEUMONIA PAGE ----------------
if page == "🫁 Pneumonia":
    st.markdown('<h1 class="title">🫁 Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.write("Upload a chest X-ray (.jpg / .png). If model predicts Pneumonia, fill vitals for WHO-based emergency triage.")
    uploaded_file = st.file_uploader("Upload Chest X-ray:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_tensor = preprocess_image(image).to(device)

        model = load_pneumonia_model()
        classes = PNEUMONIA_CLASSES
        target_layer = model.features[-1]

        with st.spinner("Analyzing image..."):
            heatmap, pred_class = generate_gradcam_image(model, image_tensor, target_layer)
            prediction = classes[pred_class]

        # Styled result card
        if prediction == "Pneumonia":
            st.markdown('<div style="background:#ff6b6b;padding:12px;border-radius:10px;color:white;"><b>⚠️ Pneumonia Detected</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#4CAF50;padding:12px;border-radius:10px;color:white;"><b>✅ No Pneumonia (Normal)</b></div>', unsafe_allow_html=True)

        # Show Grad-CAM and tabs
        img_np = np.array(image)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        tab1, tab2 = st.tabs(["📈 Prediction", "🧠 Explainability"])
        with tab1:
            st.write(f"**Model prediction:** {prediction}")
            # Quick metrics placeholder
            st.metric("Model Confidence (approx)", "—")  # you can add real softmax if desired
            st.write("**Patient Vitals / Triage**")
            # Form for vitals
            with st.form("vitals_form_pneu"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    temp = st.number_input("Body Temperature (°C)", 34.0, 42.0, 37.0)
                    spo2 = st.number_input("Oxygen Saturation (SpO₂ %)", 50, 100, 98)
                with col2:
                    rr = st.number_input("Respiratory Rate (breaths/min)", 5, 60, 20)
                    pulse = st.number_input("Heart Rate (bpm)", 30, 180, 80)
                with col3:
                    sbp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
                    dbp = st.number_input("Diastolic BP (mmHg)", 30, 130, 80)
                    age = st.number_input("Age (years)", 0, 120, 30)

                submitted = st.form_submit_button("Evaluate Severity")

            if submitted:
                emergency_flags = []
                if spo2 < 90:
                    emergency_flags.append("🔴 Low Oxygen (SpO₂ < 90%)")
                if rr > 30 and age >= 5:
                    emergency_flags.append("🔴 Rapid Breathing (RR > 30)")
                if sbp < 90 or dbp <= 60:
                    emergency_flags.append("🔴 Low Blood Pressure")
                if temp > 39 or temp < 35:
                    emergency_flags.append("🟠 Abnormal Temperature")
                if pulse > 120:
                    emergency_flags.append("🟠 Tachycardia (High Pulse)")
                if age < 5 and rr > 40:
                    emergency_flags.append("🔴 Child Respiratory Distress")

                # Metrics row
                c1, c2, c3 = st.columns(3)
                c1.metric("Temperature (°C)", f"{temp}")
                c2.metric("SpO₂ (%)", f"{spo2}")
                c3.metric("RR (breaths/min)", f"{rr}")

                c4, c5, c6 = st.columns(3)
                c4.metric("Pulse (bpm)", f"{pulse}")
                c5.metric("BP (mmHg)", f"{sbp}/{dbp}")
                c6.metric("Age", f"{age}")

                # Oxygen gauge
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = spo2,
                    title = {'text': "SpO₂"},
                    gauge = {'axis': {'range': [50, 100]},
                             'bar': {'color': "red" if spo2 < 90 else "green"},
                             'steps': [
                                 {'range': [50, 90], 'color': "#ffcccc"},
                                 {'range': [90, 100], 'color': "#d4f7d4"}
                             ]}
                ))
                st.plotly_chart(fig_g, use_container_width=True)

                if emergency_flags:
                    st.error("🚨 Emergency referral required based on WHO criteria!")
                    st.write("**Reasons:**")
                    for flag in emergency_flags:
                        st.write(f"- {flag}")
                else:
                    st.success("✅ Stable condition — No immediate emergency detected.")

                # Patient summary card
                st.markdown("### Patient Summary")
                st.markdown(f"""
                    - **Prediction:** {prediction}  
                    - **Temp:** {temp} °C  
                    - **SpO₂:** {spo2} %  
                    - **RR:** {rr} /min  
                    - **Pulse:** {pulse} bpm  
                    - **BP:** {sbp}/{dbp} mmHg  
                    - **Age:** {age} years
                """)

        with tab2:
            st.subheader("Grad-CAM Overlay")
            st.image(overlay, use_column_width=True)
            st.caption("Red regions indicate areas the model focused on.")
            st.write("⚠️ Explainability is approximate. Use with clinical judgement.")

# ---------------- ECG PAGE ----------------
if page == "⚡ ECG":
    st.markdown('<h1 class="title">⚡ ECG MI Detection</h1>', unsafe_allow_html=True)
    st.write("Upload 12-lead ECG (.csv or .npy). App will predict MI and provide cardiac vitals triage tools.")
    uploaded_file = st.file_uploader("Upload ECG (.csv or .npy):", type=["csv", "npy"])

    if uploaded_file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if uploaded_file.name.endswith(".npy"):
                arr = np.load(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                try:
                    arr = np.loadtxt(uploaded_file, delimiter=",", skiprows=1)
                except ValueError:
                    uploaded_file.seek(0)
                    arr = np.loadtxt(uploaded_file, delimiter=",")
            else:
                arr = None
        except Exception as e:
            st.error(f"Error reading ECG: {e}")
            arr = None

        if arr is not None:
            st.write(f"📏 Raw ECG shape: {arr.shape}")
            ecg_tensor = preprocess_ecg_from_array(arr)
            st.write(f"🔧 Processed shape: {tuple(ecg_tensor.shape)}")

            ecg_model = load_ecg_model()
            importance, pred = generate_ecg_saliency(ecg_model, ecg_tensor, device)
            pred_label = ECG_CLASSES[pred]
            if pred_label == "Myocardial Infarction (MI)":
                st.markdown('<div style="background:#ff6b6b;padding:12px;border-radius:10px;color:white;"><b>⚠️ MI Detected</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:#4CAF50;padding:12px;border-radius:10px;color:white;"><b>✅ No MI (Other)</b></div>', unsafe_allow_html=True)

            # Lead 1 for plotting & animation
            lead1 = ecg_tensor.detach().cpu().numpy()[0, 0, :]
            # Static saliency plot
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(lead1, linewidth=1)
            ax.imshow(cv2.resize(importance.reshape(1, -1), (len(lead1), 60)),
                      cmap="hot", aspect="auto", alpha=0.6,
                      extent=[0, len(lead1), lead1.min() - 0.3, lead1.max() + 0.3])
            ax.set_title(f"Lead 1 with Saliency — {pred_label}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

            # Tabs for Prediction & Explainability
            tab1, tab2 = st.tabs(["📈 Prediction", "🧠 Explainability"])
            with tab1:
                st.write(f"**Model prediction:** {pred_label}")

                # Card for quick metrics
                c1, c2 = st.columns(2)
                c1.metric("Prediction", f"{pred_label}")
                c2.metric("Processed length", f"{lead1.shape[0]} samples")

                # ECG playback animation toggle
                animate = st.checkbox("Play ECG Animation (simulate live)", value=False)
                playback_speed = st.slider("Playback speed (ms per chunk)", 10, 200, 40)

                if animate:
                    placeholder = st.empty()
                    chunk = 250  # how many samples to show per frame
                    chart = placeholder.line_chart(lead1[:chunk])
                    start = 0
                    # animate by adding slices
                    for i in range(chunk, len(lead1), int(chunk/4)):
                        next_slice = lead1[i:i+int(chunk/4)]
                        if next_slice.size == 0:
                            break
                        chart.add_rows(next_slice.reshape(-1, 1))
                        time.sleep(playback_speed / 1000.0)
                    placeholder.empty()
                else:
                    st.line_chart(lead1[:1000])

                # MI emergency vitals form
                st.write("### Cardiac Vitals & Symptoms")
                with st.form("ecg_vitals_form_main"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        hr = st.number_input("Heart Rate (bpm)", 20, 220, 80)
                        spo2_ecg = st.number_input("Oxygen Saturation (SpO₂ %)", 50, 100, 98)
                    with col2:
                        sbp_ecg = st.number_input("Systolic BP (mmHg)", 50, 240, 120)
                        dbp_ecg = st.number_input("Diastolic BP (mmHg)", 30, 160, 80)
                    with col3:
                        temp_ecg = st.number_input("Body Temperature (°C)", 34.0, 42.0, 37.0)
                        chest_pain = st.number_input("Chest Pain Duration (minutes)", 0, 300, 15)
                        age_ecg = st.number_input("Age (years)", 1, 120, 50)

                    st.markdown("#### Additional Symptoms (check if present)")
                    colA, colB, colC = st.columns(3)
                    with colA:
                        sob = st.checkbox("Shortness of Breath")
                    with colB:
                        sweating = st.checkbox("Excessive Sweating")
                    with colC:
                        nausea = st.checkbox("Nausea / Vomiting")

                    submitted_ecg = st.form_submit_button("Evaluate Cardiac Severity")

                if submitted_ecg:
                    emergency_flags_ecg = []
                    if hr < 50 or hr > 120:
                        emergency_flags_ecg.append("🔴 Abnormal Heart Rate (Brady/Tachycardia)")
                    if sbp_ecg < 90 or sbp_ecg > 180:
                        emergency_flags_ecg.append("🔴 Critical Blood Pressure")
                    if spo2_ecg < 90:
                        emergency_flags_ecg.append("🔴 Low Oxygen (SpO₂ < 90%)")
                    if chest_pain > 20:
                        emergency_flags_ecg.append("🔴 Prolonged Chest Pain (>20 min)")
                    if temp_ecg > 38:
                        emergency_flags_ecg.append("🟠 Fever - may worsen cardiac condition")
                    if sob or sweating or nausea:
                        emergency_flags_ecg.append("🔴 Classic MI Symptoms (SOB / Sweating / Nausea)")
                    if age_ecg > 45:
                        emergency_flags_ecg.append("🟠 Higher risk due to age")

                    # Metrics + gauges
                    colg1, colg2, colg3 = st.columns(3)
                    colg1.metric("HR (bpm)", f"{hr}")
                    colg2.metric("SpO₂ (%)", f"{spo2_ecg}")
                    colg3.metric("BP", f"{sbp_ecg}/{dbp_ecg} mmHg")

                    fig_sp = go.Figure(go.Indicator(mode="gauge+number",
                                                   value=spo2_ecg,
                                                   title={'text': "SpO₂"},
                                                   gauge={'axis': {'range': [50, 100]},
                                                          'bar': {'color': "red" if spo2_ecg < 90 else "green"}}))
                    st.plotly_chart(fig_sp, use_container_width=True)

                    if emergency_flags_ecg:
                        st.error("🚨 Emergency referral required based on cardiac criteria!")
                        st.write("**Reasons:**")
                        for flag in emergency_flags_ecg:
                            st.write(f"- {flag}")
                    else:
                        st.success("✅ Stable condition — No immediate emergency detected.")

                    # Patient summary
                    st.markdown("### Patient Summary")
                    st.markdown(f"""
                        - **ECG Prediction:** {pred_label}  
                        - **HR:** {hr} bpm  
                        - **SpO₂:** {spo2_ecg} %  
                        - **BP:** {sbp_ecg}/{dbp_ecg} mmHg  
                        - **Chest Pain:** {chest_pain} minutes  
                        - **Age:** {age_ecg}
                    """)

            with tab2:
                st.subheader("Explainability & Saliency")
                st.write("Heatmap shows which time regions influenced the model (averaged across leads).")
                # Render static saliency map
                sal_map = cv2.resize(importance.reshape(1, -1), (len(lead1), 60))
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                ax2.imshow(sal_map, aspect="auto", cmap="hot")
                ax2.set_yticks([])
                ax2.set_xlabel("Time")
                st.pyplot(fig2)
                st.write("⚠️ Explainability is approximate. Deploy only with clinical oversight.")

# ---------------- RETINA PAGE ----------------
if page == "👁 Retina":
    st.markdown('<h1 class="title">👁 Retina — Diabetic Retinopathy</h1>', unsafe_allow_html=True)
    st.write("Upload fundus image (.jpg / .png).")
    uploaded_file = st.file_uploader("Upload fundus image:", type=["jpg", "jpeg", "png"], key="retina_uploader")

    if uploaded_file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Fundus Image", use_column_width=True)
        image_tensor = preprocess_image(image).to(device)

        model = load_retina_model()
        classes = RETINA_CLASSES
        target_layer = model.layer4[-1]

        with st.spinner("Analyzing image..."):
            heatmap, pred_class = generate_gradcam_image(model, image_tensor, target_layer)
            prediction = classes[pred_class]

        # Result card
        if pred_class == 0:
            st.markdown('<div style="background:#4CAF50;padding:12px;border-radius:10px;color:white;"><b>✅ No Diabetic Retinopathy</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#ffb74d;padding:12px;border-radius:10px;color:#333;"><b>⚠️ DR Stage Detected</b></div>', unsafe_allow_html=True)

        # Grad-CAM overlay
        img_np = np.array(image)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        st.subheader("Grad-CAM")
        st.image(overlay, use_column_width=True)
        st.caption("Use this with clinical judgement.")

# ---------------- ABOUT PAGE ----------------
if page == "ℹ️ About":
    st.markdown('<h1 class="title">About — AI Health Analyzer</h1>', unsafe_allow_html=True)
    st.write("""
    **Medical DL Suite** upgraded — now with:
    - Multi-page layout (Home, Pneumonia, ECG, Retina)  
    - Grad-CAM explainability for image models  
    - WHO/AHA-based vitals checks & triage prompts  
    - ECG animation & saliency visualization  
    - Metric cards & Plotly gauges

    ⚠️ **Important:** This app is decision-support only. Final diagnosis and management must be done by qualified clinicians. Use private secure environment for patient data.
    """)
    st.write("Paths (update if needed):")
    st.write(f"- BASE_PATH = `{BASE_PATH}`")
    st.write(f"- ECG_MODEL_PY = `{ECG_MODEL_PY}`")
