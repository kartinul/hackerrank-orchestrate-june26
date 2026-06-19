import os
import json
import re
import time
from google import genai
from typing import Dict, Any

MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

def get_api_keys() -> list[str]:
    """Extract all available GEMINI_KEY_* from environment."""
    keys = []
    for env_var in sorted(os.environ.keys()):
        if env_var.startswith("GEMINI_KEY_"):
            val = os.environ[env_var]
            if val:
                keys.append(val.strip())
    return keys

def call_vlm(prompt: str, image_paths: list[str]) -> str:
    """
    Calls the Vision-Language Model with failover across API keys and Model versions.
    Exhausts all keys for gemini-3.5-flash, then falls back to 3.0-flash, etc.
    Uploads images via the File API within the active API key context.
    """
    keys = get_api_keys()
    if not keys:
        raise ValueError("No API keys found. Please set GEMINI_KEY_1, GEMINI_KEY_2, etc. in your environment variables.")

    for model_index, model_name in enumerate(MODELS):
        for key_index, key in enumerate(keys):
            print(f"Status: Attempting {model_name} (Model {model_index+1}/{len(MODELS)}) using API Key {key_index+1}/{len(keys)} (prefix: {key[:4]}...)")
            client = genai.Client(api_key=key)
            uploaded_files = []
            try:
                # Upload files to the current API key's project space
                for path in image_paths:
                    path = path.strip()
                    if not path:
                        continue
                        
                    # Adjust path if running from code/ but path assumes root execution
                    if path.startswith('images/'):
                        adj_path = '../dataset/' + path
                    elif path.startswith('dataset/'):
                        adj_path = '../' + path
                    else:
                        adj_path = path

                    if os.path.exists(adj_path):
                        uploaded = client.files.upload(file=adj_path)
                        uploaded_files.append(uploaded)
                    elif os.path.exists(path):
                        uploaded = client.files.upload(file=path)
                        uploaded_files.append(uploaded)
                    else:
                        print(f"Warning: Image file not found at {path} or {adj_path}")
                        
                # Assemble request payload
                content = [prompt] + uploaded_files
                
                # Make generation request enforcing JSON output
                response = client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error with model {model_name} and key {key[:4]}...: {error_msg}")
                if "503" in error_msg or "UNAVAILABLE" in error_msg:
                    print(f"Model {model_name} is overloaded. Skipping remaining keys for this model.")
                    break
                time.sleep(1) # brief pause before retry
            finally:
                # Cleanup uploaded files to prevent quota bloat
                for uploaded_file in uploaded_files:
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception:
                        pass
                        
    raise RuntimeError("All models and API keys exhausted. Check your limits and keys.")

def parse_json_response(response_text: str) -> dict:
    """Safely extracts and parses JSON from the LLM response text."""
    if not response_text:
        return None
        
    try:
        # Strip markdown json block if model returned it despite mime_type
        json_pattern = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL)
        match = json_pattern.search(response_text)
        if match:
            clean_json = match.group(1)
        else:
            clean_json = response_text.strip()
            
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}\nRaw Text:\n{response_text}")
        return None
