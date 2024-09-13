import os
from dotenv import load_dotenv

import numpy as np
import requests

from webcamgpt.utils import compose_payload

load_dotenv(dotenv_path='C:/Users/yufei/Programming/AICORE/WebCamGPT/webcamGPT/.env') #change the PATH
API_KEY = os.getenv('OPENAI_API_KEY')


class OpanAIConnector:

    def __init__(self, api_key: str = API_KEY):
        if api_key is None:
            raise ValueError("API_KEY is not set")
        self.api_key = api_key

    def simple_prompt(self, images: list, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Use compose_payload to send multiple images with the prompt
        payload = compose_payload(images=images, prompt=prompt)
        
        try:
            # Send the request to the OpenAI API
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=payload)

            # Check for any errors in the response
            response_json = response.json()
            if 'error' in response_json:
                raise Exception(f"API Error: {response_json['error']['message']}")

            # Return the content of the first choice
            return response_json['choices'][0]['message']['content']

        except KeyError as e:
            # Print or log the full response for debugging
            print(f"KeyError: {e}, Response: {response_json}")
            raise
        except Exception as e:
            # Handle other exceptions (e.g., network issues, invalid payload)
            print(f"An error occurred: {e}")
            raise


