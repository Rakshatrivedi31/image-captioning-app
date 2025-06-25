!pip install transformers gradio
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Caption function
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Gradio interface
iface = gr.Interface(fn=generate_caption, 
                     inputs=gr.Image(type="pil"), 
                     outputs="text", 
                     title="Image Caption Generator",
                     description="Upload an image to generate a caption using BLIP Transformer")

iface.launch()
