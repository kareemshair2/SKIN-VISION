"""
Generator module for creating multimodal medical responses using active LLM providers.

This module dynamically selects and invokes the configured LLM (Google Gemini, 
OpenAI/OpenRouter, Anthropic Claude, or DeepSeek) to generate dermatological assessments 
based on user queries, retrieved medical context, and uploaded skin images.
"""

import os
from PIL import Image
from src.config import (
    ACTIVE_PROVIDER,
    GOOGLE_MODEL,
    OPENAI_MODEL,
    ANTHROPIC_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
)

# ============================================================
# 1. Global Client Initialization
# ============================================================
GOOGLE_CLIENT = None
OPENAI_CLIENT = None
ANTHROPIC_CLIENT = None
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CLIENT = None

# ============================================================
# 2. Conditional Provider Initialization
# ============================================================
if ACTIVE_PROVIDER == "google" and GOOGLE_API_KEY:
    from google import genai
    GOOGLE_CLIENT = genai.Client(api_key=GOOGLE_API_KEY)

elif ACTIVE_PROVIDER == "openai" and OPENAI_API_KEY:
    from openai import OpenAI
    # Configure OpenAI client to point to OpenRouter base URL
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

elif ACTIVE_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    import anthropic
    ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

elif ACTIVE_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
    from openai import OpenAI
    DEEPSEEK_CLIENT = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# ============================================================
# 3. Main Response Generation Dispatcher
# ============================================================

def generate_medical_response(payload: dict) -> str:
    """
    Orchestrates the generation of a medical response.
    """
    user_query = payload.get("user_query", "")
    context = payload.get("medical_context", "")
    image_path = payload.get("image_path")

    prompt = f"""
    You are an expert AI medical assistant specializing in dermatology. 
    Analyze the user's condition using the provided image and the official medical context below.
    
    CRITICAL GUIDELINES:
    1. Rely heavily on the provided medical context to explain potential conditions.
    2. Examine the image features carefully if provided.
    3. Always include a clear medical disclaimer stating that this is an educational tool 
       and does not replace professional medical diagnosis.
    
    Official Medical Context:
    {context}
    
    User Query / Description:
    {user_query}
    """

    try:
        if ACTIVE_PROVIDER == "google":
            return _generate_with_google(prompt, image_path)
        
        if ACTIVE_PROVIDER == "openai":
            return _generate_with_openai(prompt, payload.get("image_base64"))
        
        if ACTIVE_PROVIDER == "anthropic":
            return _generate_with_anthropic(prompt, payload.get("image_base64"))
        
        if ACTIVE_PROVIDER == "deepseek":
            return _generate_with_deepseek(prompt, payload.get("image_base64"))

        return "Error: Invalid or unconfigured AI provider in settings."
            
    except (RuntimeError, ValueError, Exception) as e:
        return f"An error occurred during response generation: {str(e)}"


# ============================================================
# 4. Provider-Specific Generation Handlers
# ============================================================

def _generate_with_google(prompt: str, image_path: str) -> str:
    if not GOOGLE_CLIENT:
        return "Error: Google Gemini client is not initialized."

    contents = [prompt]
    if image_path and os.path.exists(image_path):
        contents.append(Image.open(image_path))

    response = GOOGLE_CLIENT.models.generate_content(
        model=GOOGLE_MODEL,
        contents=contents
    )
    return response.text


def _generate_with_openai(prompt: str, image_base64: str) -> str:
    """
    Generates a response using OpenRouter (via OpenAI-compatible endpoint).
    """
    if not OPENAI_CLIENT:
        return "Error: OpenAI/OpenRouter client is not initialized."

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if image_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    response = OPENAI_CLIENT.chat.completions.create(
        model=OPENAI_MODEL,  # Will use the model specified in .env (e.g., openai/gpt-4o)
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content


def _generate_with_anthropic(prompt: str, image_base64: str) -> str:
    if not ANTHROPIC_CLIENT:
        return "Error: Anthropic client is not initialized."

    content_list = []
    if image_base64:
        content_list.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64
            }
        })
    content_list.append({"type": "text", "text": prompt})

    response = ANTHROPIC_CLIENT.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": content_list}]
    )
    return response.content[0].text


def _generate_with_deepseek(prompt: str, image_base64: str) -> str:
    if not DEEPSEEK_CLIENT:
        return "Error: DeepSeek client is not initialized."

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if image_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    response = DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content