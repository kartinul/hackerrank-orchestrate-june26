import os
import requests
from dotenv import load_dotenv

load_dotenv()

def list_models():
    key = os.environ.get("GEMINI_KEY_1")
    if not key:
        print("Error: GEMINI_KEY_1 is missing.")
        return

    print("Fetching models via ModelService.ListModels...\n")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        for model in data.get('models', []):
            name = model.get('name', 'Unknown')
            methods = model.get('supportedGenerationMethods', [])
            print(f"Model: {name}")
            print(f"Methods: {methods}\n")
    else:
        print("Failed to fetch models:")
        print(f"Status {response.status_code}: {response.text}")

if __name__ == "__main__":
    list_models()
