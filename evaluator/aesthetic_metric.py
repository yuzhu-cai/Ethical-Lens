import os
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from urllib.request import urlretrieve

class AestheticMetric():
	"""
	Aesthetic Predictor based on LAION dataset
	GitHub: https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb

	"""

	def __init__(self, cfg):
		self.cache_folder = cfg['cache_folder']
		self.aesthetic_model = self.get_aesthetic_model(clip_model=cfg['model'])
		self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14' if cfg['model']=='vit_l_14' else 'ViT-B-32', pretrained='openai')
		self.device = cfg['device']
		self.aesthetic_model = self.aesthetic_model.to(self.device)
		self.model = self.model.to(self.device)

	def get_aesthetic_model(self, clip_model="vit_l_14"):
    
		cache_folder = self.cache_folder
		path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
		if not os.path.exists(path_to_model):
			os.makedirs(cache_folder, exist_ok=True)
			url_model = (
				"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
			)
			urlretrieve(url_model, path_to_model)
		if clip_model == "vit_l_14":
			m = nn.Linear(768, 1)
		elif clip_model == "vit_b_32":
			m = nn.Linear(512, 1)
		else:
			raise ValueError()
		s = torch.load(path_to_model)
		m.load_state_dict(s)
		m.eval()
		return m


	def calculate(self, data):
		image_ = Image.open(data['image']['path'])
		image_ = self.preprocess(image_).unsqueeze(0).to(self.device)
		with torch.no_grad():
			image_features = self.model.encode_image(image_)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			prediction = self.aesthetic_model(image_features)
			return {'aesthetic': prediction.item()}
		
	
	def verify(self, img_path):
		import time
		start_time = time.time()
		image_ = Image.open(img_path)
		image_ = self.preprocess(image_).unsqueeze(0).to(self.device)
		with torch.no_grad():
			image_features = self.model.encode_image(image_)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			prediction = self.aesthetic_model(image_features)
			print('Time cost: ', time.time()-start_time)
			return {'aesthetic': prediction.item()}


if __name__ == "__main__":
	# In the terminal, using the folloing command:
	# >>> CUDA_VISIBLE_DEVICES=6,7 python evaluator/aesthetic_metric.py
	cfg = {
		'model': "vit_l_14",
		'device': 'cuda:0',
		'cache_folder': "/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/pretrain_model/emb_reader/"
	}
	metric = AestheticMetric(cfg)
	print(metric.verify('/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/data/images/0a1a2d23-6e76-45ad-8177-aff945161575.jpg'))
	
	print(metric.verify('/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/data/images/0a9f46f3-e36b-4442-b105-5c43d636d04b.jpg'))
	import pdb; pdb.set_trace()