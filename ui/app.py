"""
Streamlit Frontend for Sarcasm Detection System
Beautiful web interface for sarcasm detection using FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import io
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np

# Configure page
st.set_page_config(
    page_title="Sarcasm Detection System",
    page_icon="🙄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        margin-bottom: 2rem;
    }
    .prediction-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sarcastic {
        border-color: #E74C3C;
        background-color: #FADBD8;
    }
    .not-sarcastic {
        border-color: #27AE60;
        background-color: #D5F4E6;
    }
    .metric-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498DB;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Optional[Dict]:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_model_info() -> Optional[Dict]:
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def predict_single_text(text: str) -> Dict:
    """Get sarcasm prediction for single text"""
    try:
        payload = {"text": text}
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def predict_batch_texts(texts: List[str], process_text: bool = True) -> Dict:
    """Get sarcasm predictions for multiple texts"""
    try:
        payload = {"texts": texts, "process_text": process_text}
        response = requests.post(f"{API_BASE_URL}/predict/batch", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def display_prediction_result(result: Dict, show_processing_details: bool = False):
    """Display individual prediction result with styling"""
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        return
    
    # Determine styling based on prediction
    is_sarcastic = "sarcastic" in result["gemini_validation"].lower()
    emoji = "🙄" if is_sarcastic else "😊"
    label_text = "SARCASTIC" if is_sarcastic else "NOT SARCASTIC"
    box_class = "sarcastic" if is_sarcastic else "not-sarcastic"
    
    # Create prediction display
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h4>{emoji} {label_text}</h4>
        <p><strong>Final Output:</strong> {result['gemini_validation']}</p>
    </div>
    """, unsafe_allow_html=True)


def create_batch_visualization(batch_result: Dict):
    """Create visualizations for batch prediction results"""
    if "error" in batch_result:
        st.error(f"Error: {batch_result['error']}")
        return
    
    predictions = batch_result["predictions"]
    summary = batch_result["summary"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", summary["total_texts"])
    with col2:
        st.metric("Sarcastic", summary["sarcastic_count"])
    with col3:
        st.metric("Not Sarcastic", summary["not_sarcastic_count"])
    with col4:
        st.metric("Avg Confidence", f"{summary['average_confidence']:.1%}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of sarcasm distribution
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=["Sarcastic", "Not Sarcastic"],
                values=[summary["sarcastic_count"], summary["not_sarcastic_count"]],
                hole=0.4,
                colors=["#E74C3C", "#27AE60"]
            )
        ])
        fig_pie.update_layout(title="Sarcasm Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution histogram
        confidences = [p["confidence"] for p in predictions]
        fig_hist = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Distribution",
            labels={"x": "Confidence", "y": "Count"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    results_df = pd.DataFrame([
        {
            "Text": p["text"][:50] + "..." if len(p["text"]) > 50 else p["text"],
            "Label": "🙄 Sarcastic" if p["label"] == "sarcasm" else "😊 Not Sarcastic",
            "Probability": f"{p['sarcasm_probability']:.1%}",
            "Confidence": f"{p['confidence']:.1%}"
        }
        for p in predictions
    ])
    
    st.dataframe(results_df, use_container_width=True)


class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.audio_buffer = io.BytesIO()
        self.is_recording = False

    def recv(self, frame):
        if self.is_recording:
            self.audio_buffer.write(frame.to_ndarray().tobytes())
        return frame

    def start_recording(self):
        self.is_recording = True
        self.audio_buffer = io.BytesIO()

    def stop_recording(self):
        self.is_recording = False
        audio_data = sr.AudioData(self.audio_buffer.getvalue(), 48000, 2)
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🙄 Sarcasm Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7F8C8D; font-size: 1.2em;">AI-powered sarcasm detection with hybrid NLP model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Info")
        
        # API Health Check
        health = check_api_health()
        if health and health.get("status") == "healthy":
            st.success("✅ API is running")
            
            # Model info
            model_info = get_model_info()
            if model_info:
                st.markdown("**📊 Model Performance:**")
                st.write(f"• Type: {model_info['model_type']}")
                st.write(f"• F1-Score: {model_info['f1_score']:.1%}")
                st.write(f"• Features: {model_info['feature_counts']['total_hybrid_features']:,}")
        else:
            st.error("❌ API not available")
            st.info("Make sure to start the API: `uvicorn api.main:app --reload`")
            st.stop()
        
        st.markdown("---")
        st.markdown("**🎯 Model Features:**")
        st.write("• TF-IDF text analysis")
        st.write("• Contextual linguistics")  
        st.write("• Sentiment analysis")
        st.write("• Hyperbole detection")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Text", "🎤 Voice Input", "📁 Batch Analysis", "📖 About"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Input form
        with st.form("single_text_form"):
            text_input = st.text_area(
                "Enter text to analyze for sarcasm:",
                height=100,
                placeholder="Type your text here... (e.g., 'Oh great, another Monday morning!')"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                process_text = st.checkbox("Apply text preprocessing", value=True)
            with col2:
                submit_button = st.form_submit_button("🔍 Analyze", type="primary")
        
        if submit_button and text_input:
            with st.spinner("Analyzing text..."):
                result = predict_single_text(text_input)
                display_prediction_result(result)
        
        elif submit_button:
            st.warning("Please enter some text to analyze.")
        
        # Example texts
        st.subheader("💡 Try These Examples:")
        
        examples = [
            "What a wonderful day to be stuck in traffic!",
            "I absolutely love doing laundry on weekends.",
            "Thank you for the helpful feedback.",
            "Perfect timing for the server to crash!",
            "This meeting could definitely not have been an email."
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i % len(cols)]:
                if st.button(f"📝 Example {i+1}", key=f"example_{i}"):
                    result = predict_single_text(example)
                    st.write(f"**Text:** {example}")
                    display_prediction_result(result)

    with tab2:
        st.header("Analyze Voice Input")
        webrtc_ctx = webrtc_streamer(key="speech-to-text", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioProcessor)

        if webrtc_ctx.audio_processor:
            if st.button("Start Recording"):
                webrtc_ctx.audio_processor.start_recording()
                st.write("Recording...")

            if st.button("Stop Recording"):
                text = webrtc_ctx.audio_processor.stop_recording()
                st.write(f"Recognized Text: {text}")
                with st.spinner("Analyzing text..."):
                    prediction_result = predict_single_text(text)
                    display_prediction_result(prediction_result)

    with tab3:
        st.header("Batch Text Analysis")
        
        # File upload option
        st.subheader("📁 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file (CSV or TXT)",
            type=["csv", "txt"],
            help="CSV: Must have a 'text' column. TXT: One text per line."
        )
        
        # Manual text input option
        st.subheader("✏️ Manual Input")
        manual_texts = st.text_area(
            "Or enter multiple texts (one per line):",
            height=150,
            placeholder="Enter each text on a new line...\nExample line 1\nExample line 2"
        )
        
        # Processing options
        col1, col2 = st.columns([3, 1])
        with col1:
            batch_process_text = st.checkbox("Apply text preprocessing", value=True, key="batch_process")
        with col2:
            analyze_button = st.button("📊 Analyze Batch", type="primary")
        
        if analyze_button:
            texts_to_analyze = []
            
            # Process uploaded file
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts_to_analyze.extend(df['text'].dropna().astype(str).tolist())
                        else:
                            st.error("CSV file must have a 'text' column")
                    
                    elif uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        texts_to_analyze.extend([line.strip() for line in content.split('\n') if line.strip()])
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            
            # Process manual input
            if manual_texts:
                texts_to_analyze.extend([line.strip() for line in manual_texts.split('\n') if line.strip()])
            
            # Analyze if we have texts
            if texts_to_analyze:
                # Limit batch size
                if len(texts_to_analyze) > 100:
                    st.warning(f"Limiting analysis to first 100 texts (uploaded: {len(texts_to_analyze)})")
                    texts_to_analyze = texts_to_analyze[:100]
                
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    batch_result = predict_batch_texts(texts_to_analyze, batch_process_text)
                    
                    st.success(f"✅ Analysis complete!")
                    create_batch_visualization(batch_result)
                    
                    # Download results option
                    if "predictions" in batch_result:
                        results_df = pd.DataFrame([
                            {
                                "text": p["text"],
                                "original_text": p.get("original_text", ""),
                                "label": p["label"],
                                "sarcasm_probability": p["sarcasm_probability"],
                                "confidence": p["confidence"]
                            }
                            for p in batch_result["predictions"]
                        ])
                        
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download Results (CSV)",
                            csv_data,
                            "sarcasm_detection_results.csv",
                            "text/csv"
                        )
            else:
                st.warning("Please upload a file or enter texts to analyze.")
    
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 What is Sarcasm Detection?
        
        This AI system automatically detects sarcasm in text using advanced natural language processing techniques.
        It's particularly useful for:
        
        - **Social Media Analysis** - Understanding customer sentiment beyond surface-level text
        - **Content Moderation** - Identifying potentially negative content hidden behind sarcasm
        - **Customer Service** - Detecting frustrated customers who express dissatisfaction sarcastically
        - **Market Research** - Getting true sentiment from product reviews and feedback
        
        ### 🤖 How It Works
        
        Our system uses a **hybrid model** that combines:
        
        1. **TF-IDF Text Analysis** - Traditional text vectorization for word patterns
        2. **Contextual Features** - Advanced linguistic analysis including:
           - Sentiment polarity variation
           - Incongruity detection (positive words + negative context)
           - Hyperbole patterns (excessive punctuation, intensifiers)
           - Lexical complexity metrics
        
        ### 📊 Performance
        
        - **Model Type:** Hybrid Logistic Regression
        - **F1-Score:** 66.9%
        - **Training Data:** 78,619 texts from news headlines and social media
        - **Features:** 10,018 total (10,000 TF-IDF + 18 contextual)
        
        ### 💡 Tips for Best Results
        
        - **Complete sentences** work better than fragments
        - **Context matters** - the system looks for contradictory sentiment signals
        - **Punctuation patterns** like "!!" and "..." are important indicators
        - **Enable preprocessing** for consistent results
        
        ### 🛠️ Technical Stack
        
        - **Backend:** FastAPI with hybrid ML model
        - **Frontend:** Streamlit web interface
        - **ML Libraries:** Scikit-learn, NLTK, TextBlob
        - **Model Training:** Jupyter notebooks with comprehensive data analysis
        """)


if __name__ == "__main__":
    main()
