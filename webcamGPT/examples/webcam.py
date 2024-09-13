import os
import cv2
import uuid

import gradio as gr
import numpy as np

import webcamgpt

MARKDOWN = """
# webcamGPT

This is a demo of webcamGPT, a tool that allows you to chat with video using GPT-4. 
"""

connector = webcamgpt.OpanAIConnector()

# Save image to disk and return the file path
def save_image(image: np.ndarray) -> str:
    os.makedirs("data", exist_ok=True)
    image_path = os.path.join("data", f"{uuid.uuid4()}.jpeg")
    cv2.imwrite(image_path, image)
    return image_path


# Process image, update history, and send the prompt to GPT-4
def respond(image: np.ndarray, prompt: str, chat_history):
    global image_history
    
    # Flip, convert, and save the image
    image = cv2.cvtColor(np.fliplr(image), cv2.COLOR_RGB2BGR)
    image_path = save_image(image)
    
    # Update image history (keep last 20 images)
    image_history.append(image)
    image_history = image_history[-20:]
    
    # Send the prompt and image history to GPT-4
    response = connector.simple_prompt(images=image_history, prompt=prompt)
    
    # Update chat history
    chat_history.extend([((image_path,), None), (prompt, response)])
    
    return "", chat_history


# This function clears the chatbot and resets the image history
def clear_history():
    global image_history
    image_history = []
    return "", []



# Initialize image history as a global list to track history of images
image_history = [] 

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True)
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            message = gr.Textbox()
            clear_button = gr.Button("Clear")

    # Clear button functionality to clear chatbot and image history
    clear_button.click(clear_history, [], [message, chatbot])
    
    # Message submit functionality to process image and prompt and update chatbot
    message.submit(respond, [webcam, message, chatbot], [message, chatbot])

demo.launch(debug=False, show_error=True)
