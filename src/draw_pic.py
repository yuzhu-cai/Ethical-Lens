from diffusers import DiffusionPipeline, StableDiffusionXLInstructPix2PixPipeline, StableDiffusionPipeline
import torch
from PIL import Image

text2img_path = "/dev/shm/data/shengyin/pretrain_model/stabilityai--stable-diffusion-xl-base-1.0"
text2img = DiffusionPipeline.from_pretrained(text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
prompt = "zombie protestor with sign 3 5 mm "
revision = "protestor with sign"

img1 = text2img(prompt).images[0]
img1.save('origin.jpg')

img2 = text2img(revision).images[0]
img2.save('post.jpg')