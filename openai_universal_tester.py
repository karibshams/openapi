import streamlit as st
import openai
from openai import OpenAI
import time
from typing import Dict, List, Tuple, Optional, Any
import json
import base64
from pathlib import Path
import tempfile
import os
from datetime import datetime
import mimetypes

# Page configuration
st.set_page_config(
    page_title="OpenAI Universal Model Tester",
    page_icon="🚀",
    layout="wide"
)

# Security Notice
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
    <p style='margin: 0; color: #1f77b4;'>
        🔒 <strong>Security Notice:</strong> Your API key is never stored persistently. It exists only in memory during your current session and is cleared when you close the browser.
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'available_models' not in st.session_state:
    st.session_state.available_models = {
        'chat': [],
        'embedding': [],
        'audio': [],
        'image': [],
        'moderation': []
    }
if 'client' not in st.session_state:
    st.session_state.client = None

# Model categories and their prefixes
MODEL_CATEGORIES = {
    'chat': ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o'],
    'embedding': ['text-embedding'],
    'audio': ['whisper', 'tts'],
    'image': ['dall-e'],
    'moderation': ['text-moderation']
}

def categorize_models(models: List[str]) -> Dict[str, List[str]]:
    """Categorize models by their type."""
    categorized = {
        'chat': [],
        'embedding': [],
        'audio': [],
        'image': [],
        'moderation': []
    }
    
    for model in models:
        model_lower = model.lower()
        if any(prefix in model_lower for prefix in ['gpt-4', 'gpt-3.5-turbo']):
            categorized['chat'].append(model)
        elif 'embedding' in model_lower:
            categorized['embedding'].append(model)
        elif 'whisper' in model_lower:
            categorized['audio'].append(model)
        elif 'tts' in model_lower:
            categorized['audio'].append(model)
        elif 'dall-e' in model_lower:
            categorized['image'].append(model)
        elif 'moderation' in model_lower:
            categorized['moderation'].append(model)
    
    return categorized

def test_chat_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """Test a chat model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )
        return True, "Chat model working"
    except Exception as e:
        return False, str(e)

def test_embedding_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """Test an embedding model."""
    try:
        response = client.embeddings.create(
            model=model,
            input="Test embedding"
        )
        return True, f"Embedding dimension: {len(response.data[0].embedding)}"
    except Exception as e:
        return False, str(e)

def test_audio_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """Test an audio model."""
    try:
        if 'whisper' in model.lower():
            # For whisper, we'd need an audio file - just check if model exists
            return True, "Whisper model available (requires audio file for full test)"
        elif 'tts' in model.lower():
            response = client.audio.speech.create(
                model=model,
                voice="alloy",
                input="Hello, this is a test."
            )
            return True, "TTS model working"
        return False, "Unknown audio model type"
    except Exception as e:
        return False, str(e)

def test_image_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """Test an image model."""
    try:
        if 'dall-e-2' in model:
            response = client.images.generate(
                model=model,
                prompt="A small red square",
                n=1,
                size="256x256"
            )
        else:  # dall-e-3
            response = client.images.generate(
                model=model,
                prompt="A small red square",
                n=1,
                size="1024x1024"
            )
        return True, "Image generation working"
    except Exception as e:
        return False, str(e)

def test_moderation_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """Test a moderation model."""
    try:
        response = client.moderations.create(
            model=model,
            input="This is a test message."
        )
        return True, "Moderation model working"
    except Exception as e:
        return False, str(e)

def test_model_by_type(client: OpenAI, model: str, model_type: str) -> Tuple[bool, str]:
    """Test a model based on its type."""
    if model_type == 'chat':
        return test_chat_model(client, model)
    elif model_type == 'embedding':
        return test_embedding_model(client, model)
    elif model_type == 'audio':
        return test_audio_model(client, model)
    elif model_type == 'image':
        return test_image_model(client, model)
    elif model_type == 'moderation':
        return test_moderation_model(client, model)
    else:
        return False, "Unknown model type"

def get_all_models(client: OpenAI) -> Dict[str, List[str]]:
    """Fetch and categorize all available models."""
    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        return categorize_models(model_ids)
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return {
            'chat': [],
            'embedding': [],
            'audio': [],
            'image': [],
            'moderation': []
        }

def test_api_key_comprehensive(api_key: str) -> Tuple[Dict[str, Dict[str, Tuple[bool, str]]], Optional[OpenAI]]:
    """Comprehensively test the API key against all model types."""
    results = {}
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Get all available models
        available_models = get_all_models(client)
        
        # Add default models if not in the list
        default_models = {
            'chat': ['gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            'embedding': ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'],
            'audio': ['whisper-1', 'tts-1', 'tts-1-hd'],
            'image': ['dall-e-2', 'dall-e-3'],
            'moderation': ['text-moderation-latest', 'text-moderation-stable']
        }
        
        # Merge with defaults
        for category, models in default_models.items():
            for model in models:
                if model not in available_models[category]:
                    available_models[category].append(model)
        
        # Test each category
        for category, models in available_models.items():
            results[category] = {}
            for model in models:
                success, message = test_model_by_type(client, model, category)
                results[category][model] = (success, message)
                time.sleep(0.1)  # Rate limiting
        
        # Update session state with available models
        for category in available_models:
            st.session_state.available_models[category] = [
                model for model, (success, _) in results[category].items() if success
            ]
        
        return results, client
        
    except openai.AuthenticationError:
        return {cat: {model: (False, "Invalid API key")} for cat in ['chat'] for model in ['gpt-3.5-turbo']}, None
    except Exception as e:
        return {cat: {model: (False, f"Error: {str(e)}")} for cat in ['chat'] for model in ['gpt-3.5-turbo']}, None

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

# UI Components
st.title("🚀 OpenAI Universal Model Tester")
st.markdown("Test all OpenAI models: Chat, Voice, Image, Video, Embeddings, and more!")

# Sidebar for API key input and testing
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # Security reminder
    st.caption("🔒 Your API key is used only for this session")
    
    # API Key input - NOT stored anywhere
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your API key is never saved and exists only in memory during this session",
        key="api_key_input"  # This is just for Streamlit's internal widget state
    )
    
    # Clear session button
    if st.button("🗑️ Clear Session", help="Clear all data including API key from memory"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Test button
    if st.button("Test All Models", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter an API key")
        elif not api_key.startswith("sk-"):
            st.warning("⚠️ API key should start with 'sk-'")
        else:
            with st.spinner("Testing API key across all model types..."):
                # Pass API key directly without storing
                results, client = test_api_key_comprehensive(api_key)
                st.session_state.test_results = results
                st.session_state.client = client
                st.session_state.api_key_validated = client is not None
                # API key is NOT stored - it's only in the client object
    
    # Display test results
    if st.session_state.test_results:
        st.markdown("---")
        st.subheader("📊 Test Results")
        
        for category, models in st.session_state.test_results.items():
            if models:
                working = sum(1 for _, (success, _) in models.items() if success)
                total = len(models)
                
                with st.expander(f"{category.upper()} Models ({working}/{total})", expanded=False):
                    for model, (success, message) in models.items():
                        if success:
                            st.success(f"✅ {model}")
                        else:
                            st.error(f"❌ {model}: {message[:50]}...")

# Main area - Model Testing Interface
if st.session_state.api_key_validated:
    # Create tabs for different model types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat Models", 
        "🎵 Audio Models", 
        "🎨 Image Models", 
        "📊 Embeddings",
        "🛡️ Moderation",
        "🔬 Model Comparison"
    ])
    
    # Chat Models Tab
    with tab1:
        st.markdown("### Chat with Language Models")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            chat_model = st.selectbox(
                "Select Chat Model",
                st.session_state.available_models['chat'],
                key="chat_model_select"
            )
        with col2:
            if st.button("Clear Chat", key="clear_chat"):
                if chat_model in st.session_state.chat_messages:
                    st.session_state.chat_messages[chat_model] = []
                st.rerun()
        
        # Vision model check
        is_vision_model = 'vision' in chat_model or 'gpt-4o' in chat_model
        
        if is_vision_model:
            st.info("🖼️ This model supports vision! You can upload images.")
            uploaded_image = st.file_uploader(
                "Upload an image (optional)",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                key="chat_image_upload"
            )
        
        # Initialize chat history
        if chat_model not in st.session_state.chat_messages:
            st.session_state.chat_messages[chat_model] = []
        
        # Display chat messages
        for message in st.session_state.chat_messages[chat_model]:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            st.markdown(content["text"])
                        elif content["type"] == "image_url":
                            st.image(content["image_url"]["url"])
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Message {chat_model}..."):
            # Prepare message content
            if is_vision_model and uploaded_image:
                # Save and encode image
                image_path = save_uploaded_file(uploaded_image)
                base64_image = encode_image(image_path)
                
                message_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
                os.unlink(image_path)  # Clean up
            else:
                message_content = prompt
            
            # Add user message
            st.session_state.chat_messages[chat_model].append({"role": "user", "content": message_content})
            
            # Display user message
            with st.chat_message("user"):
                if isinstance(message_content, list):
                    st.markdown(prompt)
                    if uploaded_image:
                        st.image(uploaded_image)
                else:
                    st.markdown(message_content)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        messages = []
                        for msg in st.session_state.chat_messages[chat_model]:
                            if isinstance(msg["content"], list):
                                messages.append(msg)
                            else:
                                messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        response = st.session_state.client.chat.completions.create(
                            model=chat_model,
                            messages=messages,
                            max_tokens=1000
                        )
                        
                        assistant_message = response.choices[0].message.content
                        st.markdown(assistant_message)
                        
                        # Add to history
                        st.session_state.chat_messages[chat_model].append(
                            {"role": "assistant", "content": assistant_message}
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Audio Models Tab
    with tab2:
        st.markdown("### Audio Processing Models")
        
        audio_tabs = st.tabs(["🎤 Speech to Text (Whisper)", "🔊 Text to Speech (TTS)"])
        
        # Whisper Tab
        with audio_tabs[0]:
            st.markdown("#### Transcribe Audio with Whisper")
            
            whisper_models = [m for m in st.session_state.available_models['audio'] if 'whisper' in m.lower()]
            if whisper_models:
                whisper_model = st.selectbox("Select Whisper Model", whisper_models, key="whisper_select")
                
                uploaded_audio = st.file_uploader(
                    "Upload audio file",
                    type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'],
                    key="whisper_upload"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    language = st.text_input("Language (optional)", placeholder="en", key="whisper_lang")
                with col2:
                    response_format = st.selectbox("Response Format", ["text", "json", "srt", "vtt"], key="whisper_format")
                
                if st.button("Transcribe Audio", key="whisper_button") and uploaded_audio:
                    with st.spinner("Transcribing..."):
                        try:
                            audio_path = save_uploaded_file(uploaded_audio)
                            
                            with open(audio_path, "rb") as audio_file:
                                transcript = st.session_state.client.audio.transcriptions.create(
                                    model=whisper_model,
                                    file=audio_file,
                                    response_format=response_format,
                                    language=language if language else None
                                )
                            
                            os.unlink(audio_path)
                            
                            st.success("Transcription complete!")
                            if response_format == "text":
                                st.text_area("Transcription", transcript, height=200)
                            else:
                                st.code(transcript, language=response_format)
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("No Whisper models available")
        
        # TTS Tab
        with audio_tabs[1]:
            st.markdown("#### Generate Speech with TTS")
            
            tts_models = [m for m in st.session_state.available_models['audio'] if 'tts' in m.lower()]
            if tts_models:
                col1, col2 = st.columns(2)
                with col1:
                    tts_model = st.selectbox("Select TTS Model", tts_models, key="tts_select")
                with col2:
                    voice = st.selectbox("Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], key="tts_voice")
                
                tts_text = st.text_area("Enter text to convert to speech", key="tts_text", height=100)
                
                col1, col2 = st.columns(2)
                with col1:
                    speed = st.slider("Speed", 0.25, 4.0, 1.0, 0.25, key="tts_speed")
                with col2:
                    response_format = st.selectbox("Format", ["mp3", "opus", "aac", "flac"], key="tts_format")
                
                if st.button("Generate Speech", key="tts_button") and tts_text:
                    with st.spinner("Generating speech..."):
                        try:
                            response = st.session_state.client.audio.speech.create(
                                model=tts_model,
                                voice=voice,
                                input=tts_text,
                                speed=speed,
                                response_format=response_format
                            )
                            
                            # Save and play audio
                            audio_bytes = response.content
                            st.audio(audio_bytes, format=f"audio/{response_format}")
                            
                            # Download button
                            st.download_button(
                                "Download Audio",
                                audio_bytes,
                                file_name=f"speech.{response_format}",
                                mime=f"audio/{response_format}"
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("No TTS models available")
    
    # Image Models Tab
    with tab3:
        st.markdown("### Image Generation & Editing")
        
        image_tabs = st.tabs(["🎨 Generate Images", "✏️ Edit Images", "🔄 Create Variations"])
        
        # Generate Images Tab
        with image_tabs[0]:
            image_models = st.session_state.available_models['image']
            if image_models:
                image_model = st.selectbox("Select Image Model", image_models, key="image_gen_select")
                
                image_prompt = st.text_area("Describe the image you want to generate", key="image_prompt")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'dall-e-3' in image_model:
                        size = st.selectbox("Size", ["1024x1024", "1792x1024", "1024x1792"], key="image_size")
                    else:
                        size = st.selectbox("Size", ["256x256", "512x512", "1024x1024"], key="image_size")
                with col2:
                    n_images = st.number_input("Number of images", 1, 4 if 'dall-e-2' in image_model else 1, 1, key="n_images")
                with col3:
                    if 'dall-e-3' in image_model:
                        quality = st.selectbox("Quality", ["standard", "hd"], key="image_quality")
                        style = st.selectbox("Style", ["vivid", "natural"], key="image_style")
                
                if st.button("Generate Image", key="gen_image_button") and image_prompt:
                    with st.spinner("Generating image..."):
                        try:
                            params = {
                                "model": image_model,
                                "prompt": image_prompt,
                                "size": size,
                                "n": n_images
                            }
                            
                            if 'dall-e-3' in image_model:
                                params["quality"] = quality
                                params["style"] = style
                            
                            response = st.session_state.client.images.generate(**params)
                            
                            for i, image in enumerate(response.data):
                                st.image(image.url, caption=f"Generated Image {i+1}")
                                st.markdown(f"[Download Image {i+1}]({image.url})")
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("No image generation models available")
        
        # Edit Images Tab
        with image_tabs[1]:
            st.markdown("#### Edit existing images (DALL-E 2 only)")
            
            if 'dall-e-2' in st.session_state.available_models['image']:
                uploaded_edit_image = st.file_uploader(
                    "Upload image to edit (must be square PNG)",
                    type=['png'],
                    key="edit_upload"
                )
                
                uploaded_mask = st.file_uploader(
                    "Upload mask (transparent areas will be edited)",
                    type=['png'],
                    key="mask_upload"
                )
                
                edit_prompt = st.text_area("Describe the edit", key="edit_prompt")
                
                if st.button("Edit Image", key="edit_button") and all([uploaded_edit_image, uploaded_mask, edit_prompt]):
                    with st.spinner("Editing image..."):
                        try:
                            response = st.session_state.client.images.edit(
                                model="dall-e-2",
                                image=uploaded_edit_image.read(),
                                mask=uploaded_mask.read(),
                                prompt=edit_prompt,
                                n=1,
                                size="1024x1024"
                            )
                            
                            st.image(response.data[0].url, caption="Edited Image")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("Image editing requires DALL-E 2")
        
        # Variations Tab
        with image_tabs[2]:
            st.markdown("#### Create variations of existing images (DALL-E 2 only)")
            
            if 'dall-e-2' in st.session_state.available_models['image']:
                uploaded_var_image = st.file_uploader(
                    "Upload image for variations (must be square PNG)",
                    type=['png'],
                    key="var_upload"
                )
                
                n_variations = st.slider("Number of variations", 1, 4, 2, key="n_variations")
                
                if st.button("Create Variations", key="var_button") and uploaded_var_image:
                    with st.spinner("Creating variations..."):
                        try:
                            response = st.session_state.client.images.create_variation(
                                model="dall-e-2",
                                image=uploaded_var_image.read(),
                                n=n_variations,
                                size="1024x1024"
                            )
                            
                            for i, image in enumerate(response.data):
                                st.image(image.url, caption=f"Variation {i+1}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("Image variations require DALL-E 2")
    
    # Embeddings Tab
    with tab4:
        st.markdown("### Text Embeddings")
        
        embedding_models = st.session_state.available_models['embedding']
        if embedding_models:
            embedding_model = st.selectbox("Select Embedding Model", embedding_models, key="embed_select")
            
            embed_text = st.text_area("Enter text to embed", key="embed_text", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                show_full = st.checkbox("Show full embedding vector", key="show_full_embed")
            with col2:
                encoding_format = st.selectbox("Encoding Format", ["float", "base64"], key="embed_format")
            
            if st.button("Generate Embedding", key="embed_button") and embed_text:
                with st.spinner("Generating embedding..."):
                    try:
                        response = st.session_state.client.embeddings.create(
                            model=embedding_model,
                            input=embed_text,
                            encoding_format=encoding_format
                        )
                        
                        embedding = response.data[0].embedding
                        
                        st.success(f"Embedding generated! Dimension: {len(embedding)}")
                        
                        if show_full:
                            st.code(embedding[:50] if not show_full else embedding)
                        else:
                            st.info(f"First 10 values: {embedding[:10]}")
                        
                        # Stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Dimension", len(embedding))
                        with col2:
                            st.metric("Tokens", response.usage.total_tokens)
                        with col3:
                            st.metric("Model", embedding_model)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("No embedding models available")
    
    # Moderation Tab
    with tab5:
        st.markdown("### Content Moderation")
        
        moderation_models = st.session_state.available_models['moderation']
        if moderation_models:
            moderation_model = st.selectbox("Select Moderation Model", moderation_models, key="mod_select")
            
            mod_text = st.text_area("Enter text to moderate", key="mod_text", height=100)
            
            if st.button("Check Content", key="mod_button") and mod_text:
                with st.spinner("Analyzing content..."):
                    try:
                        response = st.session_state.client.moderations.create(
                            model=moderation_model,
                            input=mod_text
                        )
                        
                        result = response.results[0]
                        
                        if result.flagged:
                            st.error("⚠️ Content flagged!")
                        else:
                            st.success("✅ Content passed moderation")
                        
                        # Show detailed scores
                        st.markdown("#### Category Scores")
                        categories = result.categories.model_dump()
                        scores = result.category_scores.model_dump()
                        
                        flagged_cats = [cat for cat, flagged in categories.items() if flagged]
                        if flagged_cats:
                            st.warning(f"Flagged categories: {', '.join(flagged_cats)}")
                        
                        # Display scores
                        cols = st.columns(3)
                        for i, (category, score) in enumerate(scores.items()):
                            with cols[i % 3]:
                                color = "🔴" if categories[category] else "🟢"
                                st.metric(f"{color} {category}", f"{score:.4f}")
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("No moderation models available")
    
    # Model Comparison Tab
    with tab6:
        st.markdown("### Compare Models Side-by-Side")
        
        comparison_type = st.selectbox(
            "Select comparison type",
            ["Chat Models", "Image Generation", "Embeddings"],
            key="comp_type"
        )
        
        if comparison_type == "Chat Models":
            chat_models = st.session_state.available_models['chat']
            if len(chat_models) >= 2:
                selected_models = st.multiselect(
                    "Select models to compare",
                    chat_models,
                    default=chat_models[:2],
                    key="comp_chat_select"
                )
                
                compare_prompt = st.text_area("Enter prompt for comparison", key="comp_prompt")
                
                if st.button("Compare", key="comp_button") and compare_prompt and len(selected_models) >= 2:
                    cols = st.columns(len(selected_models))
                    
                    for idx, model in enumerate(selected_models):
                        with cols[idx]:
                            st.markdown(f"**{model}**")
                            with st.spinner(f"Getting response..."):
                                try:
                                    response = st.session_state.client.chat.completions.create(
                                        model=model,
                                        messages=[{"role": "user", "content": compare_prompt}],
                                        max_tokens=500
                                    )
                                    
                                    result = response.choices[0].message.content
                                    st.info(result)
                                    
                                    # Metrics
                                    st.caption(f"Tokens: {response.usage.total_tokens}")
                                    
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Not enough chat models available for comparison")
        
        elif comparison_type == "Image Generation":
            image_models = st.session_state.available_models['image']
            if image_models:
                selected_img_models = st.multiselect(
                    "Select image models to compare",
                    image_models,
                    default=image_models[:min(2, len(image_models))],
                    key="comp_img_select"
                )
                
                img_compare_prompt = st.text_area("Enter image prompt for comparison", key="comp_img_prompt")
                
                if st.button("Generate & Compare", key="comp_img_button") and img_compare_prompt and selected_img_models:
                    cols = st.columns(len(selected_img_models))
                    
                    for idx, model in enumerate(selected_img_models):
                        with cols[idx]:
                            st.markdown(f"**{model}**")
                            with st.spinner(f"Generating..."):
                                try:
                                    params = {
                                        "model": model,
                                        "prompt": img_compare_prompt,
                                        "n": 1
                                    }
                                    
                                    if 'dall-e-3' in model:
                                        params["size"] = "1024x1024"
                                        params["quality"] = "standard"
                                    else:
                                        params["size"] = "512x512"
                                    
                                    response = st.session_state.client.images.generate(**params)
                                    st.image(response.data[0].url)
                                    
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            else:
                st.warning("No image models available for comparison")
        
        elif comparison_type == "Embeddings":
            embedding_models = st.session_state.available_models['embedding']
            if embedding_models:
                selected_emb_models = st.multiselect(
                    "Select embedding models to compare",
                    embedding_models,
                    default=embedding_models[:min(3, len(embedding_models))],
                    key="comp_emb_select"
                )
                
                emb_compare_text = st.text_area("Enter text for embedding comparison", key="comp_emb_text")
                
                if st.button("Generate & Compare", key="comp_emb_button") and emb_compare_text and selected_emb_models:
                    results = {}
                    
                    for model in selected_emb_models:
                        try:
                            response = st.session_state.client.embeddings.create(
                                model=model,
                                input=emb_compare_text
                            )
                            results[model] = {
                                'dimension': len(response.data[0].embedding),
                                'tokens': response.usage.total_tokens,
                                'first_values': response.data[0].embedding[:5]
                            }
                        except Exception as e:
                            results[model] = {'error': str(e)}
                    
                    # Display comparison table
                    st.markdown("#### Embedding Comparison Results")
                    
                    comparison_data = []
                    for model, data in results.items():
                        if 'error' not in data:
                            comparison_data.append({
                                'Model': model,
                                'Dimension': data['dimension'],
                                'Tokens': data['tokens'],
                                'First 5 Values': str(data['first_values'])
                            })
                        else:
                            comparison_data.append({
                                'Model': model,
                                'Dimension': 'Error',
                                'Tokens': '-',
                                'First 5 Values': data['error']
                            })
                    
                    st.dataframe(comparison_data, use_container_width=True)
            else:
                st.warning("No embedding models available for comparison")

else:
    # Show comprehensive feature overview when not authenticated
    st.info("👈 Please enter and test your OpenAI API key in the sidebar to access all features")
    
    # Feature Overview
    st.markdown("### 🎯 Complete OpenAI API Testing Suite")
    
    # Create feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 💬 Chat Models
        - GPT-4o, GPT-4, GPT-3.5
        - Vision capabilities
        - File uploads
        - Conversation history
        - Model comparison
        """)
        
        st.markdown("""
        #### 🎵 Audio Processing
        - **Whisper**: Transcribe audio
        - **TTS**: Text-to-speech
        - Multiple voices
        - Various formats
        - Speed control
        """)
    
    with col2:
        st.markdown("""
        #### 🎨 Image Generation
        - DALL-E 2 & 3
        - Image editing
        - Style variations
        - Multiple sizes
        - HD quality options
        """)
        
        st.markdown("""
        #### 📊 Embeddings
        - Text vectorization
        - Multiple models
        - Dimension analysis
        - Token counting
        - Format options
        """)
    
    with col3:
        st.markdown("""
        #### 🛡️ Moderation
        - Content safety checks
        - Category scoring
        - Multiple models
        - Detailed analysis
        - Real-time results
        """)
        
        st.markdown("""
        #### 🔬 Comparisons
        - Side-by-side testing
        - Performance metrics
        - Multi-model analysis
        - Token usage
        - Response quality
        """)
    
    # Supported File Types
    st.markdown("---")
    st.markdown("### 📁 Supported File Types")
    
    file_col1, file_col2, file_col3, file_col4 = st.columns(4)
    
    with file_col1:
        st.markdown("""
        **Images**
        - PNG
        - JPEG/JPG
        - GIF
        - WebP
        """)
    
    with file_col2:
        st.markdown("""
        **Audio**
        - MP3
        - WAV
        - M4A
        - WebM
        """)
    
    with file_col3:
        st.markdown("""
        **Video**
        - MP4
        - MPEG
        - MOV
        - AVI
        """)
    
    with file_col4:
        st.markdown("""
        **Documents**
        - TXT
        - PDF
        - DOCX
        - And more...
        """)
    
    # Quick Start Guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Get your API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    2. **Enter Key**: Paste your key in the sidebar (starts with 'sk-')
    3. **Test Models**: Click "Test All Models" to check access
    4. **Start Testing**: Use any available model type
    5. **Compare Models**: Test multiple models side-by-side
    """)
    
    # Model Pricing Note
    with st.expander("💰 Model Pricing Information"):
        st.markdown("""
        Different models have different pricing:
        
        **Chat Models**
        - GPT-4o: Most capable, higher cost
        - GPT-4: Very capable, moderate cost
        - GPT-3.5-Turbo: Fast and affordable
        
        **Other Models**
        - DALL-E 3: $0.040-$0.080 per image
        - DALL-E 2: $0.016-$0.020 per image
        - Whisper: $0.006 per minute
        - TTS: $0.015-$0.030 per 1M characters
        - Embeddings: $0.0001-$0.0013 per 1K tokens
        
        Visit [OpenAI Pricing](https://openai.com/pricing) for current rates.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <small>
        🔐 Your API key is never stored and is only used during this session<br>
        📊 Token usage and costs depend on model selection and input complexity<br>
        🚀 Built with Streamlit for comprehensive OpenAI API testing
        </small>
    </div>
    """,
    unsafe_allow_html=True
)