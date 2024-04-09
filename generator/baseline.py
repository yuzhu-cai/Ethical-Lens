import warnings
warnings.filterwarnings("ignore")

import os
import time
import random
import torch
import numpy as np

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from torchvision import transforms
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

import sys
sys.path.append('./')
from common.random_seed import prepare_seed

def dummy(images, **kwargs): 
	return images, None
def dummy_safe(images, **kwargs): 
	return images, []

class Baseline():
	"""
	A construction for inference
	"""

	def __init__(self, cfg, image_save_path='test'):
		# Specify the relative GPU index. If the CUDA_VISIBLE_DEVICES=2,5,7, then
		# cuda:0 for GPU-2, cuda:1 for GPU-5.
		rgi = cfg['relative_gpu_index']
		self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')
		
		# load parameters
		self.resolution = cfg['resolution']
		self.text2img_path = cfg['text2img_path']
		if 'image_dir' in cfg.keys():
			self.image_save_path = cfg['image_dir']
		else:
			self.image_save_path = image_save_path
		self.image_transform = transforms.ToTensor()
		# load model
		self.text2img = self.load_models(self.text2img_path)


	def load_models(
		self,
		text2img_path="/GPFS/public/align_yuzhucai/huggingface/hub/stabilityai--stable-diffusion-xl-base-1.0", 
	):
		
		# text2img = DiffusionPipeline.from_pretrained(text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		if "v1-5" in text2img_path:
			text2img = StableDiffusionPipeline.from_pretrained(text2img_path, torch_dtype=torch.float16).to(self.device)
			text2img.safety_checker = dummy
		else:
			text2img = DiffusionPipeline.from_pretrained(text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		

		return text2img
	
	def generate(
			self,
			data
	):
		data['status'] = 'SUCCESS'
		image_id = '0000001'
		prompt_id = data['id']

		# generate an image by stable-diffusion-model-xl-1.0
		prepare_seed(self.random_seed)
		prompt = data['prompt']
		if 'negative_prompt' in data:
			raw_image = self.text2img(prompt=prompt, negative_prompt=data['negative_prompt'], num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
		else:
			if "sld" in self.text2img_path:
				raw_image = self.text2img(prompt=prompt, **SafetyConfig.MAX, num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
			else:
				raw_image = self.text2img(prompt=prompt, num_inference_steps=50, requires_safety_checker=False).images[0].resize((self.resolution, self.resolution))
		
		save_path = os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_raw_image.jpg")
		raw_image.thumbnail((512, 512))
		raw_image.save(save_path)
		torch.cuda.empty_cache()

		is_black = all(all(pixel == 0 for pixel in rgb) for rgb in raw_image.getdata())
		if is_black:
			data['status'] = 'FAIL'
			return
		
		raw_image_np = np.array(raw_image)
		data['id'] = image_id
		data['image'] = {
			'path': save_path,
			'data': raw_image_np,
			'data_torch': (self.image_transform(raw_image_np)*255).byte(),
			'width': raw_image_np.shape[0],
			'height': raw_image_np.shape[1],
		}
		return


	def inference(self, data, id):
		
		if 'random_seed' in data.keys():
			self.random_seed = data['random_seed']
		else:
			random.seed()
			self.random_seed = random.randint(0, 1000000)
		data.update({
			'status': 'SUCCESS',
			'time': 0.0,
			'random_seed': self.random_seed
		})
		start_time = time.time()
		# Generate the image and update the status simultaneously.
		self.generate(data)

		end_time = time.time()
		data['time'] = (end_time - start_time)*1000

		return data

class Baseline_Safe():
	"""
	A construction for inference
	"""

	def __init__(self, cfg, image_save_path='test'):
		# Specify the relative GPU index. If the CUDA_VISIBLE_DEVICES=2,5,7, then
		# cuda:0 for GPU-2, cuda:1 for GPU-5.
		rgi = cfg['relative_gpu_index']
		self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')
		
		# load parameters
		self.resolution = cfg['resolution']
		self.text2img_path = cfg['text2img_path']
		if 'image_dir' in cfg.keys():
			self.image_save_path = cfg['image_dir']
		else:
			self.image_save_path = image_save_path
		self.image_transform = transforms.ToTensor()
		# load model
		self.text2img = self.load_models(self.text2img_path)


	def load_models(
		self,
		text2img_path="/GPFS/public/align_yuzhucai/huggingface/hub/stabilityai--stable-diffusion-xl-base-1.0", 
	):
		
		text2img = StableDiffusionPipelineSafe.from_pretrained(text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		text2img.safety_checker = dummy_safe
		return text2img
	
	def generate(
			self,
			data
	):
		data['status'] = 'SUCCESS'
		image_id = '0000001'
		prompt_id = data['id']

		# generate an image by stable-diffusion-model-xl-1.0
		prepare_seed(self.random_seed)
		prompt = data['prompt']
		if 'negative_prompt' in data:
			raw_image = self.text2img(prompt=prompt, negative_prompt=data['negative_prompt'], num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
		else:
			raw_image = self.text2img(prompt=prompt, **SafetyConfig.MAX, num_inference_steps=50).images[0].resize((self.resolution, self.resolution))

		save_path = os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_raw_image.jpg")
		raw_image.thumbnail((512, 512))
		raw_image.save(save_path)
		torch.cuda.empty_cache()

		is_black = all(all(pixel == 0 for pixel in rgb) for rgb in raw_image.getdata())
		if is_black:
			data['status'] = 'FAIL'
			return
		
		raw_image_np = np.array(raw_image)
		data['id'] = image_id
		data['image'] = {
			'path': save_path,
			'data': raw_image_np,
			'data_torch': (self.image_transform(raw_image_np)*255).byte(),
			'width': raw_image_np.shape[0],
			'height': raw_image_np.shape[1],
		}
		return


	def inference(self, data, id):
		
		if 'random_seed' in data.keys():
			self.random_seed = data['random_seed']
		else:
			random.seed()
			self.random_seed = random.randint(0, 1000000)
		data.update({
			'status': 'SUCCESS',
			'time': 0.0,
			'random_seed': self.random_seed
		})
		start_time = time.time()
		# Generate the image and update the status simultaneously.
		self.generate(data)

		end_time = time.time()
		data['time'] = (end_time - start_time)*1000

		return data

	

if __name__ == "__main__":
	# In the terminal, using the following command:
	# >>> CUDA_VISIBLE_DEVICES=5,7 python generator/baseline.py
	cfg = {
		'text2img_path' : "/dssg/home/acct-umjpyb/umjpyb/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79",
		'image_dir': './',
		'negative_prompt': "NSFW, nude",

		'resolution' : 512,
		'relative_gpu_index' : 0
	}
	data_example = {
		'prompt': 'A cut dog',
		'prompt_id': 'test'
	}
	if 'negative_prompt' in cfg:
		data_example['negative_prompt'] = cfg['negative_prompt']
	generator = Baseline(cfg)
	generator.inference(data_example, 'test')
	# import pdb; pdb.set_trace()
