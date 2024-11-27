# TEXT_TO_IMAGE
# Text to Image Generation using Stable Diffusion

This project demonstrates how to generate images from text prompts using the **Stable Diffusion** model. The model takes in natural language descriptions and generates corresponding images based on the input. This project uses **Hugging Face's Diffusers** library and the **Stable Diffusion** model to perform text-to-image generation.

## Features

- **Generate Images**: Convert text descriptions (e.g., "an astronaut in space") into visual representations.
- **Model**: Uses **Stable Diffusion v2** from Hugging Face for image generation.
- **Customizable**: You can modify parameters like image size, inference steps, and guidance scale.
- **GPU Support**: Can be run on a GPU for faster image generation (by changing the device configuration).
- **Save and Display Images**: Display generated images using `matplotlib` and save them as PNG files.

## Prerequisites

Before running the project, ensure that you have the following installed:

- Python 3.7 or higher
- **PyTorch**: For working with deep learning models.
- **Transformers**: For loading pre-trained models like GPT-2 for text generation (optional).
- **Diffusers**: For the Stable Diffusion model.
- **Matplotlib**: For image visualization.

### Install Required Packages

To install the necessary libraries, run the following command:

```bash
pip install torch diffusers transformers matplotlib opencv-python
Setup
Clone this Repository:
bash
Copy code
git clone https://github.com/your-username/text-to-image.git
cd text-to-image
Hugging Face Authentication: To use Stable Diffusion and other Hugging Face models, you'll need to authenticate via your Hugging Face account.

Sign up or log in to Hugging Face.
Obtain your API token from Hugging Face Tokens.
Set your token as an environment variable or input it directly into the code.
bash
Copy code
export HF_AUTH_TOKEN="your_hugging_face_auth_token"
Alternatively, replace the token in the code directly:

python
Copy code
use_auth_token='your_hugging_face_auth_token'
Usage
1. Text-to-Image Generation
To generate an image from a text prompt, run the following in your Python environment:

python
Copy code
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

# Configuration Class
class CFG:
    device = "cpu"  # Change to "cuda" for GPU support
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
Replace "astronaut in space" with any text prompt you want to generate an image for. The generated image will be displayed and saved to the local directory as a PNG file.

2. Adjust Parameters
You can modify the following parameters in the CFG class to customize the image generation:

device: "cpu" or "cuda" for GPU support.
image_gen_steps: Number of steps for image generation (higher steps result in better images).
image_gen_guidance_scale: Controls how strongly the model follows the prompt. Higher values result in images that better match the description.
image_gen_size: Set the resolution of the generated image (e.g., (512, 512) or (400, 400)).
Example Outputs
Prompt: "Astronaut in space"

Generated Image: (Image of an astronaut floating in space)
Prompt: "A fantasy castle on a mountain"

Generated Image: (Image of a beautiful castle surrounded by mountains)
Contributing
Feel free to fork this repository and make improvements or add new features. To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-name).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Hugging Face for providing pre-trained models like Stable Diffusion.
Stable Diffusion for the image generation model.
Generated by [Your Name].

markdown
Copy code

### Key Sections in the README:

1. **Project Overview**: Describes what the project is about, its capabilities, and the technologies used.
2. **Features**: Lists the key features of the project.
3. **Prerequisites and Installation**: Provides instructions to install the necessary dependencies.
4. **Setup**: Guides users to set up the project, including authenticating with Hugging Face.
5. **Usage**: Explains how to use the script to generate images and adjust the settings.
6. **Example Outputs**: Demonstrates the kind of results users can expect.
7. **Contributing**: Encourages others to contribute to the project.
8. **License**: Information about the project's license.
9. **Acknowledgements**: Credits to the libraries and frameworks used.

### How to Use:

1. Copy this `README.md` to your repository.
2. Replace `"your-username"` with your GitHub username.
3. Add any relevant images or examples if needed.
4. Make sure to include the correct installation instructions for any additional dependencies if they are used in the code.

This should help other developers understand and use your **Text to Image** project easily!


