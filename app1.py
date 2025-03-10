import streamlit as st
import torch
import librosa
import soundfile as sf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import tempfile

# Page configuration
st.set_page_config(
    page_title="Audio to Image Converter",
    page_icon="🎵➡️🖼️",
    layout="wide"
)

# Set up cache directory for models
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

@st.cache_resource
def load_speech_model():
    """Load the speech recognition model with caching"""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

@st.cache_resource
def load_image_model():
    """Load the text-to-image model with caching"""
    try:
        if torch.cuda.is_available():
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cuda")
            # Apply optimizations
            pipe.enable_attention_slicing()
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5"
            ).to("cpu")
        st.success("✅ Image generation model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Failed to load image generation model: {str(e)}")
        return None

def transcribe_audio(audio_file, processor, speech_model, sampling_rate=16000):
    """
    Convert audio file to text using Wav2Vec2
    
    Parameters:
        audio_file: Path to audio file or audio data as numpy array
        processor: Wav2Vec2 processor
        speech_model: Wav2Vec2 model
        sampling_rate: Sample rate of the audio
        
    Returns:
        Transcribed text
    """
    try:
        # Load audio if a file path is provided
        if isinstance(audio_file, str):
            audio, _ = librosa.load(audio_file, sr=sampling_rate)
        else:
            # Assume it's already audio data
            audio = audio_file
        
        # Process audio with Wav2Vec2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = speech_model(input_values).logits
        
        # Decode the predictions to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        
        return transcription[0]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def generate_image(text_prompt, pipe, enhancement=True):
    """
    Generate image from text prompt using Stable Diffusion
    
    Parameters:
        text_prompt: Text description for the image
        pipe: StableDiffusionPipeline
        enhancement: Whether to enhance the prompt for better quality
        
    Returns:
        Generated image
    """
    try:
        # Enhance prompt for better results if requested
        if enhancement:
            prompt = f"{text_prompt}, high quality, detailed, photorealistic"
        else:
            prompt = text_prompt
        
        # Generate image
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device):
            image = pipe(prompt, num_inference_steps=30).images[0]
        
        return image
    except Exception as e:
        st.error(f"Error during image generation: {str(e)}")
        return None

def main():
    # Add a title and description
    st.title("🎵 Audio to Image Converter 🖼️")
    st.write("""
    Upload an audio file, and this app will:
    1. Transcribe the audio to text using Wav2Vec2
    2. Generate an image from the transcribed text using Stable Diffusion
    Alternatively, you can directly generate an image from a text prompt.
    """)
    
    # GPU info
    col1, col2 = st.columns(2)
    with col1:
        if torch.cuda.is_available():
            st.success(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.info(f"GPU Memory: {memory_gb:.2f} GB")
        else:
            st.warning("⚠️ No GPU detected. Running on CPU (this will be slower)")
    
    # Load models with progress indicators
    with st.spinner("Loading models... (this may take a few minutes on first run)"):
        with col2:
            progress_bar = st.progress(0)
            
            # Load the speech model
            progress_bar.progress(25)
            st.info("Loading speech recognition model...")
            processor, speech_model = load_speech_model()
            
            # Load the image generation model
            progress_bar.progress(50)
            st.info("Loading text-to-image model...")
            pipe = load_image_model()
            
            progress_bar.progress(100)
            st.success("✅ Models loaded successfully!")
    
    # Add file uploader for audio
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file (WAV or MP3)", type=["wav", "mp3"])
    
    # Add prompt enhancement option
    enhance_prompt = st.checkbox("Enhance prompt for better image quality", value=True)
    
    # Add custom prompt option
    use_custom_prompt = st.checkbox("Use custom prompt instead of transcription")
    custom_prompt = ""
    if use_custom_prompt:
        custom_prompt = st.text_input("Enter your custom prompt:", "")
    
    if uploaded_file is not None:
        # Display the uploaded audio
        st.audio(uploaded_file)
        
        # Process button
        if st.button("Process Audio"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Step 1: Transcribe audio to text
                status_text.text("Transcribing audio...")
                progress_bar.progress(25)
                
                transcription = transcribe_audio(temp_path, processor, speech_model)
                
                if transcription:
                    # Display transcription
                    st.subheader("Transcription")
                    st.write(transcription)
                    
                    progress_bar.progress(50)
                    
                    # Step 2: Generate image from text
                    status_text.text("Generating image...")
                    
                    # Use custom prompt if specified, otherwise use transcription
                    if use_custom_prompt and custom_prompt:
                        prompt_text = custom_prompt
                    else:
                        prompt_text = transcription
                    
                    image = generate_image(prompt_text, pipe, enhancement=enhance_prompt)
                    
                    if image:
                        progress_bar.progress(100)
                        status_text.text("Done!")
                        
                        # Display the generated image
                        st.subheader("Generated Image")
                        st.image(image, caption=f"Generated from: '{prompt_text}'")
                        
                        # Option to download the image
                        img_bytes = tempfile.NamedTemporaryFile(suffix=".png")
                        image.save(img_bytes.name)
                        with open(img_bytes.name, "rb") as file:
                            btn = st.download_button(
                                label="Download Image",
                                data=file,
                                file_name="generated_image.png",
                                mime="image/png"
                            )
                    else:
                        st.error("Failed to generate image.")
                else:
                    st.error("Failed to transcribe audio.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Add a new section for direct image generation
    st.subheader("Generate Image from Text Prompt")
    direct_prompt = st.text_input("Enter a text prompt to generate an image:")
    if st.button("Generate Image"):
        if direct_prompt:
            with st.spinner("Generating image..."):
                try:
                    image = generate_image(direct_prompt, pipe, enhancement=enhance_prompt)
                    st.subheader("Generated Image")
                    st.image(image, caption=f"Generated from: '{direct_prompt}'")
                    
                    # Option to download the image
                    img_bytes = tempfile.NamedTemporaryFile(suffix=".png")
                    image.save(img_bytes.name)
                    with open(img_bytes.name, "rb") as file:
                        btn = st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a text prompt to generate an image.")
    
    # Add expander with information about the models
    with st.expander("About the models"):
        st.markdown("""
        ### Speech Recognition Model
        - **Model**: Wav2Vec2 (facebook/wav2vec2-large-960h)
        - **Purpose**: Converts speech audio to text
        - **More info**: [Hugging Face Model Card](https://huggingface.co/facebook/wav2vec2-large-960h)
        
        ### Image Generation Model
        - **Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
        - **Purpose**: Generates images from text descriptions
        - **More info**: [Hugging Face Model Card](https://huggingface.co/runwayml/stable-diffusion-v1-5)
        """)

if __name__ == "__main__":
    main()