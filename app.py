import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import os
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Microplastic Analysis System", layout="wide")
st.title("Microplastic Morphology & Ecological Risk Classifier")
st.markdown("---")

# --- DATA INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []

# Scientific Impact Reference Data
IMPACT_DATA = {
    "fibre": "High risk of gastrointestinal blockage and internal knotting. Fibres often adsorb high concentrations of Polychlorinated Biphenyls (PCBs) and mimic natural prey.",
    "fragment": "Sharp edges cause internal physical trauma and lacerations. High surface area-to-volume ratio promotes the adsorption of Persistent Organic Pollutants (POPs).",
    "film": "High potential for smothering respiratory organs such as gills. The thin, flexible morphology facilitates accidental ingestion by low-trophic level organisms.",
    "pellet": "Primary industrial source (nurdles). Often contains manufacturing additives and flame retardants that bioaccumulate rapidly up the food chain.",
    "unknown": "General physical risk to marine biota. Particle requires secondary spectroscopic verification for polymer identification."
}

# --- PDF REPORT GENERATION ---
def create_pdf(class_name, size, risk, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 20, txt="Scientific Analysis Report: Microplastic Identification", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Classification: {class_name.capitalize()}", ln=True)
    pdf.cell(0, 10, txt=f"Calculated Dimension: {size:.2f} um", ln=True)
    pdf.cell(0, 10, txt=f"Ecological Threat Index: {risk}/100", ln=True)
    pdf.cell(0, 10, txt=f"AI Confidence: {confidence*100:.1f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=f"Environmental Impact Analysis: {IMPACT_DATA.get(class_name, IMPACT_DATA['unknown'])}")
    return pdf.output(dest='S').encode('latin-1')

# --- MODEL LOADING (TFLITE) ---
@st.cache_resource
def load_tflite_model():
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, 'model.tflite') 
        label_path = os.path.join(curr_dir, 'labels.txt')
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        with open(label_path, 'r') as f:
            l = f.readlines()
        return interp, l
    except Exception as e:
        st.sidebar.error(f"System Error: {e}")
        return None, None

interpreter, labels = load_tflite_model()

# --- SIDEBAR: CALIBRATION & PARAMETERS ---
st.sidebar.header("Calibration & Parameters")
site_name = st.sidebar.text_input("Sample Source Location", "Coastal Station Alpha")
ppm = st.sidebar.number_input("Pixels Per Micron (PPM)", value=1.0, min_value=0.01)
norm_mode = st.sidebar.selectbox("AI Normalization Mode", ["(x / 127.5) - 1", "x / 255.0"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

if st.sidebar.button("Reset Session History"):
    st.session_state.history = []
    st.rerun()

# --- MAIN ANALYSIS INTERFACE ---
uploaded_file = st.file_uploader("Upload Microplastic Sample Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Raw Sample Input")
        st.image(img_rgb, use_container_width=True)

    # OpenCV Image Processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Identify the largest particle in the frame
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = np.int64(cv2.boxPoints(rect))
        size_um = max(rect[1]) * ppm
        
        res_img = img_rgb.copy()
        cv2.drawContours(res_img, [box], 0, (0, 255, 0), 3)

        # AI Classification Logic
        if interpreter:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            
            # Apply Normalization
            img_array = (img_array / 127.5) - 1 if norm_mode == "(x / 127.5) - 1" else img_array / 255.0
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            index = np.argmax(prediction)
            class_name = labels[index].strip().split(' ', 1)[-1].lower()
            confidence = prediction[0][index]
        else:
            class_name, confidence = "unknown", 0.0

        with col2:
            st.subheader("Classification Output")
            st.image(res_img, caption="Object Detection Overlay", use_container_width=True)
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Morphology", class_name.capitalize())
            m_col2.metric("Size (um)", f"{size_um:.1f}")
            m_col3.metric("Confidence", f"{confidence*100:.1f}%")

        # --- CENTERED ECOLOGICAL THREAT GAUGE ---
        st.markdown("---")
        risk_weights = {"fibre": 1.5, "fragment": 1.2, "film": 1.0, "pellet": 0.8}
        risk_val = min(int((risk_weights.get(class_name, 1.0) * (1000/(size_um+1))) * 10), 100)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.subheader("Ecological Threat Assessment", anchor=False)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_val,
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, 40], 'color': "#A9DFBF"},
                        {'range': [40, 70], 'color': "#FAD7A0"},
                        {'range': [70, 100], 'color': "#E6B0AA"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # --- IMPACT SIMULATION SECTION ---
        st.subheader("Marine Population Bio-availability Projection")
        st.info(f"Scientific Impact Analysis: {IMPACT_DATA.get(class_name, IMPACT_DATA['unknown'])}")
        
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            org_affected = int((5000 / (size_um + 1)) * 10)
            st.write(f"In a standard cubic meter aquatic sample, a particle of this morphology and scale is statistically ingestible by approximately **{org_affected}** micro-organisms (Zooplankton/Larval Fish).")
        
        with col_m2:
            if st.button("Archive to Session Records"):
                st.session_state.history.append({
                    "Site": site_name, 
                    "Morphology": class_name.capitalize(),
                    "Size": size_um, 
                    "Risk_Index": risk_val
                })
                st.success("Record Successfully Archived")

        st.markdown("---")
        pdf_data = create_pdf(class_name.capitalize(), size_um, risk_val, confidence)
        st.download_button("Export Technical PDF Report", data=pdf_data, file_name=f"Microplastic_Analysis_{class_name}.pdf")

# --- BATCH TREND ANALYTICS ---
if st.session_state.history:
    st.markdown("---")
    st.header("Session Summary & Trend Analysis")
    df = pd.DataFrame(st.session_state.history)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.write("**Archived Batch Data**")
        st.dataframe(df, use_container_width=True)
    with t_col2:
        fig_pie = px.pie(df, names='Morphology', title='Distribution of Detected Morphotypes', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.write("**Threat Correlation (Size vs. Risk Index)**")
    fig_scatter = px.scatter(df, x="Size", y="Risk_Index", color="Morphology", size="Risk_Index", hover_data=['Site'])
    st.plotly_chart(fig_scatter, use_container_width=True)
