"""
Streamlit application for SKIN-VISION Multimodal Medical RAG System.
"""

import os
import streamlit as st
import requests
from datetime import datetime
from PIL import Image
from src.retriever import get_relevant_context
from src.multimodal_processor import prepare_multimodal_payload
from src.generator import generate_medical_response

# Google Sheet Web App URL (ربط جوجل شيت)
GOOGLE_SHEET_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbzK5C3xSHwghuRkvpX2Wg1c9HtWbGD0K7WyTCW5iSbIPllPBky4v5yB49h-OzHXKDACXQ/exec"

# Streamlit Page Configuration
st.set_page_config(
    page_title="SKIN-VISION | Multimodal Medical RAG",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS matching the exact design
st.markdown("""
    <style>
    .main-banner {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-banner h1 {
        color: white;
        font-size: 28px;
        margin-bottom: 5px;
    }
    .main-banner p {
        color: #e0e0e0;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # ==========================================
    # SIDEBAR (القائمة الجانبية المطابقة للصورة)
    # ==========================================
    with st.sidebar:
        st.markdown("### 🩺 SKIN-VISION")
        st.caption("AI Clinical Decision Support System")
        st.divider()
        
        st.markdown("#### ⓘ How to use:")
        st.markdown("1. Describe symptoms clearly.")
        st.markdown("2. Upload an image of the skin condition.")
        st.markdown("3. Click **Analyze Condition** to get insights.")
        
        st.divider()
        st.markdown("🔒 **Encrypted & Private Processing**")

    # ==========================================
    # MAIN CONTENT AREA
    # ==========================================
    
    # Banner Header
    st.markdown("""
        <div class="main-banner">
            <h1>🩺 SKIN-VISION Medical Assistant</h1>
            <p>Advanced AI Multimodal RAG system for accurate skin condition analysis and professional medical insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Info notice box
    st.info("ℹ️ Please provide both a detailed description of the symptoms and an image for the most accurate analysis.")

    # Input Fields Split (Text Area & File Uploader side by side)
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("**👤 Describe Your Symptoms**")
        user_query = st.text_area(
            "Symptoms description",
            placeholder="E.g., I have red patches on my face, itchy and scaly, appeared 3 days ago...",
            height=160,
            label_visibility="collapsed"
        )
        st.caption("0/1000")

    with col2:
        st.markdown("**📤 Upload Skin Image**")
        uploaded_file = st.file_uploader(
            "Upload skin condition image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.caption("Supports: JPG, JPEG, PNG (Max 10MB)")

    temp_image_path = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        os.makedirs("temp_uploads", exist_ok=True)
        temp_image_path = os.path.join("temp_uploads", uploaded_file.name)
        image.save(temp_image_path)

    # Analyze Button (Full width)
    analyze_clicked = st.button("✨ Analyze Condition", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_query and not temp_image_path:
            st.warning("Please provide a text description or upload an image to proceed.")
        else:
            with st.spinner("Retrieving medical context and analyzing with AI..."):
                query_to_search = user_query if user_query else "skin condition symptoms and treatment"
                retrieved_chunks = get_relevant_context(query_to_search, k=3)
                
                payload = prepare_multimodal_payload(
                    image_path=temp_image_path,
                    context_chunks=retrieved_chunks,
                    user_query=user_query
                )
                
                response = generate_medical_response(payload)
                
                st.subheader("📋 Medical Analysis Report")
                st.markdown(response)
                
                with st.expander("🔍 View Retrieved Medical References"):
                    for idx, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Reference {idx + 1} (Disease ID: {chunk['metadata'].get('disease_id')})**")
                        st.text(chunk["content"])
                        st.divider()

    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except OSError:
            pass

    st.divider()
    st.markdown("<p style='text-align: center; color: gray;'>Want to know more?</p>", unsafe_allow_html=True)

    # Expander Sections (مطابقة لأسفل الصورة)
    with st.expander("📖 What is SKIN-VISION?"):
        st.write("SKIN-VISION is an advanced AI-powered clinical decision support system designed to assist in preliminary dermatological assessments using Retrieval-Augmented Generation (RAG) and multimodal vision models.")

    with st.expander("🛡️ Disclaimer"):
        st.write("This tool is for educational and informational purposes only and does not replace professional medical diagnosis, advice, or treatment.")

    with st.expander("💡 Tips for Best Results"):
        st.write("- Provide clear, well-lit images of the affected skin area.\n- Be as descriptive as possible regarding duration, itchiness, and evolution of symptoms.")

    # ==========================================
    # OPTIONAL: Booking Form Section (إذا أردت الاحتفاظ به أسفل الصفحة)
    # ==========================================
    with st.expander("📅 Book a Clinic Appointment"):
        with st.form("patient_booking_form"):
            patient_name = st.text_input("Patient Full Name")
            patient_phone = st.text_input("Phone Number (for confirmation)")
            patient_email = st.text_input("Email Address (Optional)")
            
            col_date, col_time = st.columns(2)
            with col_date:
                appointment_date = st.date_input("Preferred Visit Date")
            with col_time:
                appointment_time = st.time_input("Preferred Visit Time")
                
            patient_notes = st.text_area("Additional Notes or Brief Description")
            submit_booking = st.form_submit_button("Confirm Booking Request", type="primary")
            
            if submit_booking:
                if patient_name and patient_phone:
                    payload = {
                        "name": patient_name,
                        "phone": patient_phone,
                        "email": patient_email,
                        "date": str(appointment_date),
                        "time": str(appointment_time),
                        "notes": patient_notes,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    try:
                        response = requests.post(GOOGLE_SHEET_WEB_APP_URL, json=payload)
                        if response.status_code == 200:
                            st.success(f"Thank you, **{patient_name}**! Your booking has been saved to Google Sheets.")
                        else:
                            st.error("Failed to save booking.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                else:
                    st.error("Please enter your name and phone number.")

if __name__ == "__main__":
    main()