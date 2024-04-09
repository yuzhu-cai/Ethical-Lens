import warnings
warnings.filterwarnings("ignore")

import re
import os
import time
import torch
import openai
import random
import numpy as np

from PIL import Image

from diffusers import DiffusionPipeline, StableDiffusionXLInstructPix2PixPipeline, StableDiffusionPipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TextStreamer, AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms

import sys
sys.path.append('./')
from common.random_seed import prepare_seed

from src.llm_wrapper import LocalChatLLM, OpenAILLM, LocalChatLLM_qwen
from src.classifier import safe_classifier
# from src.constants import expert_revision, openai_messages_spilit_alternate
from src.constants import expert_revision
from src.temp import openai_messages_spilit_alternate
from src.fuzz import ClipFuzz
def dummy(images, **kwargs): 
	return images, None

class sc_yi():
	"""
	A construction for inference
	"""

	def __init__(self, cfg, image_save_path='test'):
		# Specify the relative GPU index. If the CUDA_VISIBLE_DEVICES=2,5,7, then
		# cuda:0 for GPU-2, cuda:1 for GPU-5.
		rgi = cfg['relative_gpu_index']
		self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')

		self.random_seed = 0
		
		self.resolution = cfg['resolution']
		# load parameters
		self.text2img_path = cfg['text2img_path']
		self.llm_path = cfg['llm_path']
		self.clipseg_path = cfg['clipseg_path']
		self.checkpoints = cfg['checkpoints']

		if 'image_dir' in cfg.keys():
			self.image_save_path = cfg['image_dir']
		else:
			self.image_save_path = image_save_path
		self.image_transform = transforms.ToTensor()
		# load model
		if "v1-5" in self.text2img_path:
			self.text2img = StableDiffusionPipeline.from_pretrained(self.text2img_path, torch_dtype=torch.float16).to(self.device)
			self.text2img.safety_checker = dummy
		else:
			self.text2img = DiffusionPipeline.from_pretrained(self.text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		self.llm = LocalChatLLM(self.llm_path)
		self.sc = safe_classifier(self.checkpoints)
		self.clipfuzz = ClipFuzz(model_path=self.clipseg_path, dilation_radius=30, blur_radius=0)
		self.fuziness = 0.2


	def get_text(self, text):
		try:
			explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text).group(1).replace("\n", "")
		except:
			explanation = "<no explanation>"
		try:
			score = int(re.search(r'@@@ Score: (.*?)\s*@@@', text).group(1).replace("\n", ""))
		except:
			score = "<no score>"
		try:
			revision = re.search(r'@@@ Revision: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			revision = "<no revision>"
		return revision, score, explanation 


	def get_prompt(self, text):
		try:
			revision = re.search(r'@@@ Revision: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			revision = "<no revision>"
		return revision


	def generate(
		self,
		prompt, 
		data
	):
		openai_messages_spilit_alternate[-1]["content"] = prompt
		response = self.llm.generate(openai_messages_spilit_alternate)
		revision, score, explanation = self.get_text(response)
		data["yi_response"] = {
			"revision": revision,
			"score":score,
			"explanation": explanation
		}
		if score == "<no score>" or revision == '<no revision>':
			revision = prompt
			score = 0
		if score > 8 or revision == 'N/A':
			data['status'] = 'FAIL'
			return
		
		num_inference_steps = 0
		while num_inference_steps!=2:
			num_inference_steps+=1
			
			image_id = '0000001'
			prompt_id = data['id']

			# generate an image by stable-diffusion-model-xl-1.0
			prepare_seed(self.random_seed)
			raw_image = self.text2img(prompt=revision, num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
			raw_image.save(os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_raw_image.jpg"))
			torch.cuda.empty_cache()

			# generate a caption of the image
			issues = self.sc.check(raw_image)
			data[f"{str(num_inference_steps)}"] = {
				"issues": issues
			}
			# print(issues)

			if "nsfw" in issues or "politic" in issues or "culture" in issues:
				if num_inference_steps == 1:
					expert_revision[-1]["content"] = f"@@@ Prompt: {revision}\n@@@ Issues: {issues}\n@@@ Reason: This image contains {issues} issues. Please revise the prompt."
					opinion = self.llm.generate(expert_revision)
					revision = self.get_prompt(opinion)
					data[f"{str(num_inference_steps)}"]['revision'] = revision
				else:
					data['status'] = 'FAIL'
					return
			else:
				if "nude" in issues:
					raw_image = self.clipfuzz.fuzzy(raw_image, "nude body", self.fuziness)
				if "public" in issues:
					outputs = self.llm.generate([{"role": "user", "content": f"Point out the name of the public figure in this sentence: {revision}"}])
					data[f"{str(num_inference_steps)}"]['public_figure'] = outputs
					raw_image = self.clipfuzz.fuzzy(raw_image, f"{outputs}'s face", self.fuziness)
				# if "watermark" in issues:
				# 	raw_image = self.clipfuzz.fuzzy(raw_image, "watermark", self.fuziness)
				save_path = os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_new_image.jpg")
				raw_image.save(save_path)
				raw_image_np = np.array(raw_image)
				# data['id'] = image_id
				data['image'] = {
					'path': save_path,
					'data': raw_image_np,
					'data_torch': (self.image_transform(raw_image_np)*255).byte(),
					'width': raw_image_np.shape[0],
					'height': raw_image_np.shape[1],
				}
				return

		data['status'] = 'FAIL'
		return

	def inference(self, data, id):
		random.seed()
		self.random_seed = random.randint(0, 1000000)
		# self.random_seed = 1234
		data.update({
			'status': 'SUCCESS',
			'time': 0.0,
			'random_seed': self.random_seed
		})

		start_time = time.time()
		prompt = data['prompt']

		# Generate the image and update the status simultanenously.
		self.generate(prompt, data)

		end_time = time.time()
		data['time'] = (end_time - start_time)*1000

		return data


class sc_qwen():
	"""
	A construction for inference
	"""

	def __init__(self, cfg, image_save_path='test'):
		# Specify the relative GPU index. If the CUDA_VISIBLE_DEVICES=2,5,7, then
		# cuda:0 for GPU-2, cuda:1 for GPU-5.
		rgi = cfg['relative_gpu_index']
		self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')

		self.random_seed = 0
		
		self.resolution = cfg['resolution']
		# load parameters
		self.text2img_path = cfg['text2img_path']
		self.llm_path = cfg['llm_path']
		self.peft_path = cfg['peft_path']
		self.clipseg_path = cfg['clipseg_path']
		self.checkpoints = cfg['checkpoints']

		if 'image_dir' in cfg.keys():
			self.image_save_path = cfg['image_dir']
		else:
			self.image_save_path = image_save_path
		self.image_transform = transforms.ToTensor()
		# load model
		if "v1-5" in self.text2img_path:
			self.text2img = StableDiffusionPipeline.from_pretrained(self.text2img_path, torch_dtype=torch.float16).to(self.device)
			self.text2img.safety_checker = dummy
		else:
			self.text2img = DiffusionPipeline.from_pretrained(self.text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		self.llm = LocalChatLLM_qwen(self.llm_path, self.peft_path)
		self.sc = safe_classifier(self.checkpoints)
		self.clipfuzz = ClipFuzz(model_path=self.clipseg_path, dilation_radius=30, blur_radius=0)
		self.fuziness = 0.2


	def get_text(self, text):
		try:
			explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text).group(1).replace("\n", "")
		except:
			explanation = "<no explanation>"
		try:
			label = re.search(r'@@@ Label: (.*?)\s*@@@', text).group(1).replace("\n", "")
		except:
			label = "<no label>"
		try:
			text = re.search(r'@@@ Text: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			text = "<no text>"
		return text, label, explanation


	def get_prompt(self, text):
		try:
			revision = re.search(r'@@@ Text: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			revision = "<no text>"
		try:
			explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text, re.DOTALL).group(1).strip()
		except:
			explanation = "<no explanation>"
		return revision, explanation


	def generate(
		self,
		prompt, 
		data
	):
		openai_messages_spilit_alternate[-1]["content"] = prompt
		response = self.llm.generate(openai_messages_spilit_alternate)
		text, label, explanation = self.get_text(response)
		data["pre"] = {
			"response": response,
			"text": text,
			"label":label,
			"explanation": explanation
		}
		if label == "<no label>" or text == '<no text>':
			text = prompt
			label = 'K0'
		if label == 'K3' or text == 'N/A':
			data['status'] = 'FAIL'
			return
		
		num_inference_steps = 0
		while num_inference_steps!=2:
			num_inference_steps+=1
			prompt_id = data['id']

			# generate an image by stable-diffusion-model-xl-1.0
			prepare_seed(self.random_seed)
			raw_image = self.text2img(prompt=text, num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
			raw_image.save(os.path.join(self.image_save_path, f"{prompt_id}_raw_image.jpg"))
			torch.cuda.empty_cache()

			# generate a caption of the image
			issues = self.sc.check(raw_image)
			data[f"round_{str(num_inference_steps)}"] = {
				"issues": issues
			}
			# print(issues)

			if "nsfw" in issues or "politic" in issues or "culture" in issues:
				if num_inference_steps == 1:
					expert_revision[-1]["content"] = f"@@@ Prompt: {text} \n@@@ Issue: {issues}\n@@@ Reason: Due to the potential for {issues} issues with images generated from this prompt, please modify it to ensure the resulting images are non-toxic and suitable for viewing. The problem may stem from descriptions of the image's style or elements(zombies, Vampires, etc) within the image that reference toxic content. Please make the necessary adjustments."
					opinion = self.llm.generate(expert_revision)
					text, explanation = self.get_prompt(opinion)
					data[f"round_{str(num_inference_steps)}"]['opinion'] = opinion
					data[f"round_{str(num_inference_steps)}"]['text'] = text
					data[f"round_{str(num_inference_steps)}"]['explanation'] = explanation
				else:
					data['status'] = 'FAIL'
					return
			else:
				if "nude" in issues:
					raw_image = self.clipfuzz.fuzzy(raw_image, "nude body", self.fuziness)
				if "public" in issues:
					outputs = self.llm.generate([{"role": "user", "content": f"Point out the name of the public figure in this sentence: {text}"}])
					data[f"round_{str(num_inference_steps)}"]['public_figure'] = outputs
					raw_image = self.clipfuzz.fuzzy(raw_image, f"{outputs}'s face", self.fuziness)

				save_path = os.path.join(self.image_save_path, f"{prompt_id}_new_image.jpg")
				raw_image.save(save_path)
				raw_image_np = np.array(raw_image)
				# data['id'] = image_id
				data['image'] = {
					'path': save_path,
					'data': raw_image_np,
					'data_torch': (self.image_transform(raw_image_np)*255).byte(),
					'width': raw_image_np.shape[0],
					'height': raw_image_np.shape[1],
				}
				return

		data['status'] = 'FAIL'
		return


	def inference(self, data, id):
		random.seed()
		self.random_seed = random.randint(0, 1000000)
		# self.random_seed = 1234
		data.update({
			'status': 'SUCCESS',
			'time': 0.0,
			'random_seed': self.random_seed
		})

		start_time = time.time()
		prompt = data['prompt']

		# Generate the image and update the status simultanenously.
		self.generate(prompt, data)

		end_time = time.time()
		data['time'] = (end_time - start_time)*1000

		return data


class sc_gpt():
	"""
	A construction for inference
	"""

	def __init__(self, cfg, image_save_path='test'):
		# Specify the relative GPU index. If the CUDA_VISIBLE_DEVICES=2,5,7, then
		# cuda:0 for GPU-2, cuda:1 for GPU-5.
		rgi = cfg['relative_gpu_index']
		self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')

		self.random_seed = 0
		
		self.resolution = cfg['resolution']
		# load parameters
		self.text2img_path = cfg['text2img_path']
		self.clipseg_path = cfg['clipseg_path']
		self.api_key = cfg['api_key']
		self.checkpoints = cfg['checkpoints']

		if 'image_dir' in cfg.keys():
			self.image_save_path = cfg['image_dir']
		else:
			self.image_save_path = image_save_path
		self.image_transform = transforms.ToTensor()
		# load model
		self.text2img = DiffusionPipeline.from_pretrained(self.text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
		self.llm = OpenAILLM(self.api_key, model_name="gpt-4-turbo-preview")
		self.sc = safe_classifier(self.checkpoints)
		self.clipfuzz = ClipFuzz(model_path=self.clipseg_path, dilation_radius=30, blur_radius=0)
		self.fuziness = 0.2

	def get_text(self, text):
		try:
			explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text).group(1).replace("\n", "")
		except:
			explanation = "<no explanation>"
		try:
			label = re.search(r'@@@ Label: (.*?)\s*@@@', text).group(1).replace("\n", "")
		except:
			label = "<no label>"
		try:
			text = re.search(r'@@@ Text: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			text = "<no text>"
		return text, label, explanation 


	def get_prompt(self, text):
		try:
			revision = re.search(r'@@@ Revision: (.*?)$', text, re.DOTALL).group(1).strip()
		except:
			revision = "<no revision>"
		return revision

	def generate(
		self,
		prompt, 
		data
	):
		openai_messages_spilit_alternate[-1]["content"] = prompt
		response = self.llm.generate(openai_messages_spilit_alternate)
		print(response)
		text, label, explanation = self.get_text(response)
		data["yi_response"] = {
			"text": text,
			"label":label,
			"explanation": explanation
		}
		if label == "<no label>" or text == '<no text>':
			text = prompt
			label = 'K0'
		if label == 'K3':
			data['status'] = 'FAIL'
			return
		
		num_inference_steps = 0
		while num_inference_steps!=2:
			num_inference_steps+=1
			
			image_id = '0000001'
			prompt_id = data['id']

			# generate an image by stable-diffusion-model-xl-1.0
			prepare_seed(self.random_seed)
			raw_image = self.text2img(prompt=text, num_inference_steps=50).images[0].resize((self.resolution, self.resolution))
			raw_image.save(os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_raw_image.jpg"))
			torch.cuda.empty_cache()

			# generate a caption of the image
			issues, log = self.sc.check(raw_image)
			data[f"{str(num_inference_steps)}"] = {
				"issues": issues,
				"logits": log
			}
			# print(issues)

			if "nsfw" in issues or "politic" in issues or "culture" in issues:
				if num_inference_steps == 1:
					expert_revision[-1]["content"] = f"@@@ Prompt: {text}\n@@@ Issues: {issues}\n@@@ Reason: This image contains {issues} issues. Please revise the prompt."
					opinion = self.llm.generate(expert_revision)
					text = self.get_prompt(opinion)
					data[f"{str(num_inference_steps)}"]['revision'] = text
				else:
					data['status'] = 'FAIL'
					return
			else:
				if "nude" in issues:
					raw_image = self.clipfuzz.fuzzy(raw_image, "nude body", self.fuziness)
				if "public" in issues:
					outputs = self.llm.generate([{"role": "user", "content": f"Point out the name of the public figure in this sentence: {text}"}])
					data[f"{str(num_inference_steps)}"]['public_figure'] = outputs
					raw_image = self.clipfuzz.fuzzy(raw_image, f"{outputs}'s face", self.fuziness)
				save_path = os.path.join(self.image_save_path, f"{prompt_id}_{image_id}_new_image.jpg")
				raw_image.save(save_path)
				raw_image_np = np.array(raw_image)
				data['id'] = image_id
				data['image'] = {
					'path': save_path,
					# 'data': raw_image_np,
					# 'data_torch': (self.image_transform(raw_image_np)*255).byte(),
					'width': raw_image_np.shape[0],
					'height': raw_image_np.shape[1],
				}
				return

		data['status'] = 'FAIL'
		return

	def inference(self, data, id):
		random.seed()
		self.random_seed = random.randint(0, 1000000)
		# self.random_seed = 1234
		data.update({
			'status': 'SUCCESS',
			'time': 0.0,
			'random_seed': self.random_seed
		})

		start_time = time.time()
		prompt = data['prompt']

		# Generate the image and update the status simultanenously.
		self.generate(prompt, data)

		end_time = time.time()
		data['time'] = (end_time - start_time)*1000

		return data


if __name__ == "__main__":
	# In the terminal, using the following command:
	# >>> CUDA_VISIBLE_DEVICES=5,7 python generator/llava_gpt4.py
	cfg = {
		# 'llm_path' : "/home/ubuntu/yuxiwei/llm_finetune/Qwen-7B-Chat",
	# peft_path : "/home/ubuntu/yuxiwei/Qwen/output/7k_data/checkpoint-4000",
		# 'peft_path' : "/home/ubuntu/yuxiwei/Qwen/output/merged_r2/checkpoint-900",
		'text2img_path' : "/dev/shm/data/shengyin/pretrain_model/models--dreamlike-art--dreamlike-diffusion-1.0/snapshots/9fb5a6463bf79d81152e715e8d2a8b988f96c790",

		'image_dir': '/home/ubuntu/yuzhucai/ethicallens/generator/img',
		'checkpoints': "/home/ubuntu/yuzhucai/ethicallens/checkpoints/v2",
		'clipseg_path': "/home/ubuntu/DATA1/yuzhucai/prestrain_model/CIDAS--clipseg-rd64-refined",


		'resolution' : 512,
		'api_key' : '',
		'relative_gpu_index' : 0
	}
	import json
	with open('/home/ubuntu/DATA1/yuzhucai/data_prepare/tab/100/records_100.json', 'r') as file:
		data = json.load(file)

	generator = sc_gpt(cfg)
	res = []
	from tqdm import tqdm
	for i in tqdm(data):

		ret = generator.inference(i, 'test')
		print(ret)
		res.append(ret)
	with open('/home/ubuntu/yuzhucai/ethicallens/generator/res.json', 'w') as file:
		json.dump(res, file, indent=4)
	# >>> CUDA_VISIBLE_DEVICES=4,8,10 python generator/llm_sc.py