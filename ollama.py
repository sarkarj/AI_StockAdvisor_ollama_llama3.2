import requests
from requests.exceptions import RequestException

# Ollama API endpoint
API_URL = "http://localhost:11434/api/generate"

# Headers
HEADERS = {
    "Content-Type": "application/json"
}

# JSON request data
DATA = {
    "model": "llama3.2:latest",
    "prompt": "What is an LLM model?",
    "stream": False
}

def call_ollama_api(url, headers, data):
    """Send a POST request to the Ollama API and handle the response."""
    try:
        response = requests.post(url, json=data, headers=headers, timeout=100)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
        
        # Process and return JSON data if the request is successful
        json_response = response.json()
        print("Ollama Response:\n")
        print("Response type:", type(json_response))
        print(json_response.get("response", "No response field found in JSON"))
    
    except RequestException as e:
        # Catch any errors that occur during the request
        print(f"Request error: {e}")
    except ValueError:
        # Handle JSON decoding error
        print("Error: Unable to decode JSON response.")
    except KeyError:
        # Handle missing keys in the JSON response
        print("Error: 'response' field not found in JSON response.")

# Make the API call
call_ollama_api(API_URL, HEADERS, DATA)
