import streamlit as st
import openai
from openai import OpenAI
import time
from typing import Dict, List, Tuple, Optional
import json

# Page configuration
st.set_page_config(
    page_title="OpenAI API Key Tester",
    page_icon="🔑",
    layout="wide"
)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'client' not in st.session_state:
    st.session_state.client = None

def test_single_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """
    Test a single model with the provided API client.
    Returns (success: bool, message: str)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )
        return True, "Successfully connected"
    except openai.AuthenticationError:
        return False, "Authentication failed - Invalid API key"
    except openai.PermissionDeniedError:
        return False, "Permission denied - No access to this model"
    except openai.NotFoundError:
        return False, "Model not found"
    except openai.RateLimitError:
        return False, "Rate limit exceeded"
    except openai.APIError as e:
        return False, f"API error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def get_available_models(client: OpenAI) -> List[str]:
    """
    Fetch available models from OpenAI API.
    Returns list of model IDs that are GPT models.
    """
    try:
        models = client.models.list()
        gpt_models = [
            model.id for model in models.data 
            if 'gpt' in model.id.lower() and any(
                variant in model.id for variant in ['gpt-4', 'gpt-3.5-turbo']
            )
        ]
        return sorted(gpt_models)
    except Exception:
        # If we can't fetch models, return default list
        return []

def test_api_key(api_key: str) -> Dict[str, Tuple[bool, str]]:
    """
    Test the API key against multiple models.
    Returns dictionary with model names as keys and (success, message) tuples as values.
    """
    # Default models to test
    default_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    results = {}
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Try to get available models
        available_models = get_available_models(client)
        
        # Combine default and available models, removing duplicates
        models_to_test = list(set(default_models + available_models))
        models_to_test.sort()
        
        # Store available models and client in session state
        st.session_state.available_models = [
            model for model in models_to_test 
            if test_single_model(client, model)[0]
        ]
        st.session_state.client = client
        
        # Test each model
        for model in models_to_test:
            success, message = test_single_model(client, model)
            results[model] = (success, message)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
    except openai.AuthenticationError:
        # If authentication fails at client creation
        for model in default_models:
            results[model] = (False, "Invalid API key")
    except Exception as e:
        # General error
        for model in default_models:
            results[model] = (False, f"Client error: {str(e)}")
    
    return results

def chat_with_model(client: OpenAI, model: str, message: str, conversation_history: List[Dict]) -> Optional[str]:
    """
    Send a message to the specified model and return the response.
    """
    try:
        messages = conversation_history + [{"role": "user", "content": message}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# UI Components
st.title("🔑 OpenAI API Key Tester & Chat Interface")
st.markdown("Test your OpenAI API key and chat with different GPT models")

# Sidebar for API key input and testing
with st.sidebar:
    st.header("API Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your API key will not be stored and is only used for testing"
    )
    
    # Test button
    if st.button("Test API Key", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter an API key")
        elif not api_key.startswith("sk-"):
            st.warning("⚠️ API key should start with 'sk-'")
        else:
            with st.spinner("Testing API key across models..."):
                st.session_state.test_results = test_api_key(api_key)
                st.session_state.api_key_validated = bool(st.session_state.available_models)
    
    # Display test results in sidebar
    if st.session_state.test_results:
        st.markdown("---")
        st.subheader("Test Results")
        
        # Count successes and failures
        successes = sum(1 for success, _ in st.session_state.test_results.values() if success)
        total = len(st.session_state.test_results)
        
        # Overall summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", total)
        with col2:
            st.metric("Access", f"{successes}/{total}")
        
        # Detailed results
        with st.expander("Model Details", expanded=False):
            for model, (success, message) in sorted(st.session_state.test_results.items()):
                if success:
                    st.success(f"✅ **{model}**")
                else:
                    st.error(f"❌ **{model}**: {message[:30]}...")
        
        # Final summary
        if successes == total:
            st.success("✅ All models accessible!")
        elif successes > 0:
            st.warning(f"⚠️ Partial access: {successes}/{total}")
        else:
            st.error("❌ No models accessible")

# Main area - Chat Interface
if st.session_state.api_key_validated and st.session_state.available_models:
    # Create tabs for testing and chatting
    tab1, tab2 = st.tabs(["💬 Chat with Models", "📊 Model Comparison"])
    
    with tab1:
        st.markdown("### Chat with Available Models")
        st.markdown("Select a model and start chatting to test its capabilities")
        
        # Model selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox(
                "Choose a model",
                st.session_state.available_models,
                help="Select from your accessible models"
            )
        with col2:
            if st.button("Clear Chat", type="secondary"):
                if selected_model in st.session_state.chat_messages:
                    st.session_state.chat_messages[selected_model] = []
                st.rerun()
        
        # Initialize chat history for selected model
        if selected_model not in st.session_state.chat_messages:
            st.session_state.chat_messages[selected_model] = []
        
        # Chat container
        chat_container = st.container()
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.chat_messages[selected_model]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Message {selected_model}..."):
            # Add user message to chat history
            st.session_state.chat_messages[selected_model].append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_model(
                        st.session_state.client,
                        selected_model,
                        prompt,
                        st.session_state.chat_messages[selected_model]
                    )
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_messages[selected_model].append({"role": "assistant", "content": response})
    
    with tab2:
        st.markdown("### Compare Model Responses")
        st.markdown("Send the same prompt to multiple models and compare their responses")
        
        # Model selection for comparison
        comparison_models = st.multiselect(
            "Select models to compare",
            st.session_state.available_models,
            default=st.session_state.available_models[:2] if len(st.session_state.available_models) >= 2 else st.session_state.available_models,
            help="Choose 2-4 models for best comparison"
        )
        
        # Comparison prompt
        comparison_prompt = st.text_area(
            "Enter a prompt to test across models",
            placeholder="Example: Explain quantum computing in simple terms",
            height=100
        )
        
        if st.button("Compare Models", type="primary", disabled=not comparison_models or not comparison_prompt):
            st.markdown("---")
            
            # Create columns for responses
            cols = st.columns(len(comparison_models))
            
            for idx, model in enumerate(comparison_models):
                with cols[idx]:
                    st.markdown(f"**{model}**")
                    with st.spinner(f"Getting response from {model}..."):
                        response = chat_with_model(
                            st.session_state.client,
                            model,
                            comparison_prompt,
                            []
                        )
                        
                        # Display in a contained box
                        with st.container():
                            st.info(response)
                            
                            # Token count estimation (rough)
                            token_estimate = len(response.split()) * 1.3
                            st.caption(f"~{int(token_estimate)} tokens")

else:
    # Show instructions if API key not validated
    st.info("👈 Please enter and test your OpenAI API key in the sidebar to start chatting with models")
    
    # Show some example use cases
    st.markdown("### What you can do with this tool:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🧪 Test API Access
        - Verify API key validity
        - Check model permissions
        - Identify available models
        """)
    
    with col2:
        st.markdown("""
        #### 💬 Interactive Chat
        - Chat with different models
        - Test model capabilities
        - Save conversation history
        """)
    
    with col3:
        st.markdown("""
        #### 🔍 Compare Models
        - Side-by-side responses
        - Performance comparison
        - Token usage estimates
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <small>
        💡 Tip: Different models have different capabilities and costs. GPT-4 models are more capable but slower and more expensive.<br>
        🔒 Your API key is not stored and is only used during this session.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)