"""
Medical Image Classification Streamlit Application
with Google Gemini AI Chatbot Integration
"""
import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATASETS, GEMINI_CONFIG
from streamlit_app.model_loader import ModelLoader
from streamlit_app.image_processor import process_image
from streamlit_app.gemini_chatbot import GeminiChatbot

# Page configuration
st.set_page_config(
    page_title="Medical Image Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #FF9800;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_prediction_results(prediction: dict, dataset_name: str):
    """Display prediction results in a nice format."""
    dataset_config = DATASETS[dataset_name]
    class_names = dataset_config["classes"]

    st.markdown("### Prediction Results")

    # Get predicted class and confidence
    predicted_class = prediction["predicted_class"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Predicted: {class_names[predicted_class]}</h3>
        <p class="{get_confidence_class(confidence)}">
            Confidence: {confidence:.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display all class probabilities
    st.markdown("#### Class Probabilities")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.write(f"{class_name}: {prob:.1%}")

    # Store prediction for chatbot context
    st.session_state.last_prediction = {
        "dataset": dataset_name,
        "predicted_class": class_names[predicted_class],
        "confidence": confidence,
        "all_probabilities": dict(zip(class_names, [float(p) for p in probabilities]))
    }


def classification_page():
    """Main classification page."""
    st.markdown('<h1 class="main-header">Medical Image Classification</h1>', unsafe_allow_html=True)

    # Sidebar for model selection
    st.sidebar.markdown("## Model Selection")

    # Dataset selection
    dataset_options = {
        "kidney_cancer": "Kidney Cancer (CT)",
        "cervical_cancer": "Cervical Cancer (Microscopy)",
        "alzheimer": "Alzheimer's Disease (MRI)",
        "covid19": "COVID-19 (X-Ray)",
        "pneumonia": "Pneumonia (X-Ray)",
        "tuberculosis": "Tuberculosis (X-Ray)",
        "monkeypox": "Monkeypox (Skin)",
        "malaria": "Malaria (Microscopy)",
        "bone_shadow": "Bone Shadow Suppression (X-Ray)"
    }

    selected_dataset = st.sidebar.selectbox(
        "Select Condition",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x]
    )

    # Display dataset info
    dataset_config = DATASETS[selected_dataset]
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"**Classes:** {', '.join(dataset_config['classes'])}")
    st.sidebar.write(f"**Image Size:** {dataset_config['image_size']}")

    # Load model
    model_loaded = st.session_state.model_loader.load_model(selected_dataset)

    if not model_loaded:
        st.warning(f"Model for {dataset_options[selected_dataset]} not found. "
                  f"Please ensure '{dataset_config['model_file']}' exists in the saved_models directory.")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")

        uploaded_file = st.file_uploader(
            "Choose a medical image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an image for classification"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Classify button
            if st.button("Classify Image", type="primary"):
                if model_loaded:
                    with st.spinner("Analyzing image..."):
                        # Process and predict
                        processed_image = process_image(
                            image,
                            dataset_name=selected_dataset,
                            target_size=dataset_config["image_size"]
                        )

                        prediction = st.session_state.model_loader.predict(
                            selected_dataset,
                            processed_image
                        )

                        if prediction:
                            with col2:
                                display_prediction_results(prediction, selected_dataset)
                else:
                    st.error("Please load a model first.")

    # Information box
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>Disclaimer:</strong> This is an AI-assisted tool for educational and research purposes only.
        It should not be used as a substitute for professional medical diagnosis.
        Always consult qualified healthcare providers for medical decisions.
    </div>
    """, unsafe_allow_html=True)


def chatbot_page():
    """Gemini AI Chatbot page."""
    st.markdown('<h1 class="main-header">Medical AI Assistant</h1>', unsafe_allow_html=True)

    # Initialize chatbot
    st.sidebar.markdown("## Chatbot Settings")

    api_key = st.sidebar.text_input(
        "Enter Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio"
    )

    if api_key:
        if st.session_state.chatbot is None or st.session_state.chatbot.api_key != api_key:
            st.session_state.chatbot = GeminiChatbot(api_key)
            st.sidebar.success("Chatbot initialized!")

    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Display last prediction context if available
    if st.session_state.last_prediction:
        with st.expander("Last Classification Result (Context for AI)"):
            pred = st.session_state.last_prediction
            st.write(f"**Dataset:** {pred['dataset']}")
            st.write(f"**Predicted:** {pred['predicted_class']}")
            st.write(f"**Confidence:** {pred['confidence']:.1%}")

    # Chat interface
    st.markdown("### Chat with Medical AI Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about medical imaging or your results...")

    if user_input:
        if st.session_state.chatbot is None:
            st.error("Please enter your Gemini API key in the sidebar first.")
        else:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Get context from last prediction
            context = None
            if st.session_state.last_prediction:
                context = f"""
                The user recently classified an image with the following result:
                - Dataset: {st.session_state.last_prediction['dataset']}
                - Predicted class: {st.session_state.last_prediction['predicted_class']}
                - Confidence: {st.session_state.last_prediction['confidence']:.1%}
                - All probabilities: {st.session_state.last_prediction['all_probabilities']}
                """

            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(user_input, context)

            # Add assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()

    # Suggested questions
    st.markdown("### Suggested Questions")
    suggested = [
        "What does my classification result mean?",
        "What are the symptoms of this condition?",
        "What should I do if I have this diagnosis?",
        "How accurate is this AI classification?",
        "What are the treatment options?"
    ]

    cols = st.columns(len(suggested))
    for col, question in zip(cols, suggested):
        if col.button(question, key=question):
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            st.rerun()


def about_page():
    """About page with system information."""
    st.markdown('<h1 class="main-header">About This System</h1>', unsafe_allow_html=True)

    st.markdown("""
    ## Medical Image Classification System

    This system uses deep learning models to classify medical images across multiple conditions:

    ### Supported Conditions
    """)

    for key, value in DATASETS.items():
        st.markdown(f"""
        **{value['name']}**
        - Classes: {', '.join(value['classes'])}
        - Number of Classes: {value['num_classes']}
        - Image Size: {value['image_size']}
        - Model File: `{value['model_file']}`
        """)

    st.markdown("""
    ---

    ### Model Architectures

    The system supports multiple model architectures:

    1. **Custom CNN** - Lightweight convolutional neural network
    2. **VGG16** - Transfer learning with VGG16 pre-trained on ImageNet
    3. **ResNet50** - Transfer learning with ResNet50
    4. **InceptionV3** - Transfer learning with InceptionV3
    5. **Vision Transformer (ViT)** - Transformer-based architecture

    ---

    ### AI Chatbot

    The system integrates Google Gemini AI to provide:
    - Explanation of classification results
    - General information about medical conditions
    - Guidance on next steps (always recommending professional consultation)

    ---

    ### Disclaimer

    This tool is for **educational and research purposes only**.
    It is NOT a substitute for professional medical diagnosis or treatment.
    Always consult qualified healthcare providers for medical decisions.
    """)


def main():
    """Main application entry point."""
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Classification", "AI Assistant", "About"],
        label_visibility="collapsed"
    )

    # Display selected page
    if page == "Classification":
        classification_page()
    elif page == "AI Assistant":
        chatbot_page()
    elif page == "About":
        about_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Medical Image Classification
    Version 1.0

    Built with Streamlit and TensorFlow
    """)


if __name__ == "__main__":
    main()
