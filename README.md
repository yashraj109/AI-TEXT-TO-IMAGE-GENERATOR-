
🧠 EduSketch – AI Text-to-Image Generator

🎯 Goal: Generate educational illustrations from text prompts using Stable Diffusion.






#✅ Step 1: Install Required Libraries

!pip install diffusers transformers accelerate --upgrade
!pip install safetensors






#✅ Step 2: Import Required Libraries

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from huggingface_hub import login




#✅ Step 3: Login to Hugging Face

# Login using your Hugging Face token
login("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")  # Replace with your token




#✅ Step 4: Load the Stable Diffusion Model

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,         # Use float32 if using CPU
    use_safetensors=True
).to("cuda")  # Replace with "cpu" if GPU not available




#✅ Step 5: Educational Prompt Input

edu_prompt = input("Enter an educational topic (e.g., Water Cycle, Photosynthesis): ")






#✅ Step 6: Enhance the Prompt

enhanced_prompt = f"An educational illustration of: {edu_prompt}, highly detailed, colorful, child-friendly style, informative and clear"






#✅ Step 7: Generate the Image

generated_image = pipe(enhanced_prompt).images[0]






#✅ Step 8: Display the Image

plt.imshow(generated_image)
plt.axis('off')
plt.title(f"EduSketch Output: {edu_prompt}")
plt.show()


 
(text) 
✅ Step 9: Sample Prompts to Try

- The water cycle
- Solar system with all planets
- Photosynthesis in plants
- The life cycle of a butterfly
- Rainwater harvesting
- A child reading under a tree# AI-TEXT-TO-IMAGE-GENERATOR-
