from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

# Load base model and VAE
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
lora_model_path = "./monster_lora_model"

# Initialize the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    vae=AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16),
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# Load LoRA weights
pipe.load_lora_weights("monster_lora_model/pytorch_lora_weights.safetensors")

# Generate image
prompt = "A photo of Cthulhu in high quality with vivid coloring who is planet-sized and is spawning black holes"
negative_prompt = "blurry, bad quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

# Save the image
image.save("monster_output.png") 
