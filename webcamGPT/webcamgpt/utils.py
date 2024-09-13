import cv2
import base64
import numpy as np


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encodes a given image represented as a NumPy array to a base64-encoded string.

    Parameters:
       image (np.ndarray): A NumPy array representing the image to be encoded.

    Returns:
       str: A base64-encoded string representing the input image in JPEG format.

    Raises:
       ValueError: If the image cannot be encoded to JPEG format.
   """

    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


def compose_payload(images: list, prompt: str) -> dict:
    """
    Composes a payload dictionary with multiple base64 encoded images as history
    and a text prompt for the GPT-4 Vision model.

    Args:
        images (list): A list of images, each an np.ndarray to encode and send.
        prompt (str): The prompt text to accompany the images in the payload.

    Returns:
        dict: A structured payload for the GPT-4 Vision model, including multiple images and a prompt.
    """
    # Initialize the content list with the text prompt
    content = [{
        "type": "text",
        "text": prompt
    }]
    
    # Add each base64-encoded image to the content list
    for image in images:
        base64_image = encode_image_to_base64(image)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # Create the final payload with model, messages, and max_tokens
    return {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_tokens": 300
    }

