import os
import json
import re
import time
import threading
from collections import deque
from google import genai
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any
import constants

SeverityEnum = Enum('SeverityEnum', {s: s for s in constants.ALLOWED_SEVERITY})
ClaimStatusEnum = Enum('ClaimStatusEnum', {s: s for s in constants.ALLOWED_CLAIM_STATUS})

# Flatten dicts for Pydantic Enum validation
flat_issues = {item for sublist in constants.ALLOWED_ISSUE_TYPES.values() for item in sublist}
flat_parts = {item for sublist in constants.OBJECT_PARTS.values() for item in sublist}

IssueTypeEnum = Enum('IssueTypeEnum', {s: s for s in flat_issues})
ObjectPartEnum = Enum('ObjectPartEnum', {s: s for s in flat_parts})

class ClaimVerificationOutput(BaseModel):
    chain_of_thought: str = Field(description="Detailed, step-by-step reasoning for how you reached the final decisions. Think through the claim, history, images, and requirements here first.")
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str] = Field(description="List of risk flags chosen from the allowed list, or empty if none.")
    issue_type: list[IssueTypeEnum]
    object_part: list[ObjectPartEnum]
    claim_status: ClaimStatusEnum
    claim_status_justification: str
    supporting_image_ids: list[str] = Field(description="List of image paths supporting the decision, or empty if none.")
    valid_image: bool
    severity: SeverityEnum

MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.history = {key: deque() for key in keys}
        self.lock = threading.Lock()

    def get_available_key(self):
        """Blocks until a key is available, then returns it."""
        while True:
            with self.lock:
                now = time.time()
                for key in self.keys:
                    # Clean old requests
                    while self.history[key] and now - self.history[key][0] >= 60:
                        self.history[key].popleft()
                    
                    if len(self.history[key]) < 5:
                        self.history[key].append(now)
                        return key
            time.sleep(1)

_key_manager = None
model_cooldowns = {}
cooldown_lock = threading.Lock()

def get_api_keys() -> list[str]:
    """Extract all available GEMINI_KEY_* from environment."""
    keys = []
    for env_var in sorted(os.environ.keys()):
        if env_var.startswith("GEMINI_KEY_"):
            val = os.environ[env_var]
            if val:
                keys.append(val.strip())
    return keys

def init_key_manager():
    global _key_manager
    if _key_manager is None:
        keys = get_api_keys()
        if not keys:
            raise ValueError("No API keys found. Please set GEMINI_KEY_1, GEMINI_KEY_2, etc. in your environment variables.")
        _key_manager = KeyManager(keys)

def call_vlm(system_instruction: str, prompt: str, image_paths: list[str]) -> str:
    """
    Calls the Vision-Language Model with failover across API keys and Model versions.
    """
    init_key_manager()
    
    for model_index, model_name in enumerate(MODELS):
        # Check if model is on cooldown globally
        with cooldown_lock:
            if model_name in model_cooldowns:
                if time.time() < model_cooldowns[model_name]:
                    continue  # Skip this model, it's on cooldown
                else:
                    del model_cooldowns[model_name]
        
        # Get an available key (blocks if rate limit is reached for all keys)
        key = _key_manager.get_available_key()
        key_index = _key_manager.keys.index(key)
        
        print(f"Status: Attempting {model_name} using API Key {key_index+1}/{len(_key_manager.keys)} (prefix: {key[:4]}...)")
        
        os.environ["GEMINI_API_KEY"] = key
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
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ClaimVerificationOutput
                )
            )
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error with model {model_name} and key {key[:4]}...: {error_msg}")
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                print(f"Model {model_name} is overloaded. Skipping model globally for 60s.")
                with cooldown_lock:
                    model_cooldowns[model_name] = time.time() + 60
                continue
            time.sleep(1) # brief pause before retry
        finally:
            # Cleanup uploaded files to prevent quota bloat
            for uploaded_file in uploaded_files:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                    
    raise RuntimeError("All models exhausted or failed. Check your limits and keys.")

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
            
        parsed_obj = ClaimVerificationOutput.model_validate_json(clean_json)
        return parsed_obj.model_dump(mode='json')
    except Exception as e:
        print(f"Failed to parse JSON response: {e}\nRaw Text:\n{response_text}")
        return None
