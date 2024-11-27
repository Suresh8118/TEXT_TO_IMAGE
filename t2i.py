from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

# Configuration Class
class CFG:
    device = "cpu"  # Change to CPU for compatibility
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load the model
auth_token = os.getenv("HF_AUTH_TOKEN")  # Securely load token from environment variable
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token=auth_token, guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

# Image Generation Function
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    image = image.resize(CFG.image_gen_size)
    return image

# Generate and display image
image = generate_image("astronaut in space", image_gen_model)

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()

# Optionally save the image
image.save("generated_astronaut_in_space.png")
