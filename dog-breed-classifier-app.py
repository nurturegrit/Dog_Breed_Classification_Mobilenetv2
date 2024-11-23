import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import plotly.graph_objects as go
import io
import tf_keras
# Set page configuration
st.set_page_config(
    page_title="Dog Breed Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_breed_labels():
    """Load the mapping of class indices to breed names"""
    try:
        with open('Data/breed_labels.json', 'r') as f:
            breed_labels = json.load(f)
        return breed_labels
    except FileNotFoundError:
        st.error("breed_labels.json not found in Data directory")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding breed_labels.json")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf_keras.models.load_model(
            'Model/-1000-images-mobilenetv2-Adam',
            custom_objects={"KerasLayer": hub.KerasLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def convert_to_jpeg(image):
    """
    Convert any image format to JPEG and ensure correct color mode
    
    Args:
        image: PIL Image object
    Returns:
        PIL Image object in JPEG format
    """
    # Create a white background for transparent images
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image, mask=image.split()[1])
        image = background

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save as JPEG in memory
    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format='JPEG', quality=95)
    jpeg_buffer.seek(0)
    return Image.open(jpeg_buffer)

def preprocess_image(image):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image object
    Returns:
        numpy array ready for model input
    """
    # Resize image to match model's expected input size
    target_size = (224, 224)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def plot_top_predictions(predictions, breed_labels, top_k=10):
    """Create a bar plot of top k predictions"""
    # Get top k predictions
    top_indices = predictions[0].argsort()[-top_k:][::-1]
    top_breeds = [breed_labels[str(idx)] for idx in top_indices]
    top_probabilities = predictions[0][top_indices]
    
    # Create color array (green for highest probability, blue for others)
    colors = ['rgb(65, 105, 225)'] * top_k  # Royal Blue
    colors[0] = 'rgb(34, 139, 34)'  # Forest Green for top prediction
    
    # Create bar plot using plotly
    fig = go.Figure(data=[
        go.Bar(
            x=top_probabilities * 100,
            y=top_breeds,
            orientation='h',
            marker_color=colors,
            text=[f'{prob:.1f}%' for prob in (top_probabilities * 100)],
            textposition='auto',
        )
    ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Top 10 Predictions',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Confidence (%)',
        yaxis_title='Breed',
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def main():
    st.title("🐕 Dog Breed Classifier")
    
    # Load breed labels and model
    breed_labels = load_breed_labels()
    model = load_model()
    
    if breed_labels is None or model is None:
        st.error("Cannot continue without breed labels and model.")
        return
    
    available_breeds = sorted(breed_labels.values())

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Check Available Breeds")
        search_breed = st.text_input("Search for a breed", "")
        
        if search_breed:
            matching_breeds = [breed for breed in available_breeds 
                             if search_breed.lower() in breed.lower()]
            if matching_breeds:
                st.success(f"✓ This breed {'is' if len(matching_breeds) == 1 else 'might be'} supported!")
                if len(matching_breeds) > 1:
                    st.write("Matching breeds:")
                    for breed in matching_breeds:
                        st.write(f"- {breed}")
            else:
                st.warning("⚠️ This breed is not in our database")

    with col2:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dog image...", 
            type=["jpg", "jpeg", "png"]
        )

    # Process and display the uploaded image
    if uploaded_file is not None:
        try:
            # Load and convert image
            image = Image.open(uploaded_file)
            image = convert_to_jpeg(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions)
            predicted_breed = breed_labels[str(predicted_class)]
            confidence = float(predictions[0][predicted_class])

            # Display prediction in a nice format
            st.markdown("### Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Predicted Breed:**")
                st.markdown(f"### {predicted_breed}")
            with col2:
                st.markdown("**Confidence:**")
                st.markdown(f"### {confidence:.2%}")
            
            # Create and display the top predictions plot
            fig = plot_top_predictions(predictions, breed_labels)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Error processing the image or making prediction.")
            st.exception(e)

if __name__ == "__main__":
    main()