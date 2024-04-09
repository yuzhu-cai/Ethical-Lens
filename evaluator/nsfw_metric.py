import os
import torch
import torch.nn as nn
import autokeras as ak
import open_clip
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

class NSFWMetric():
	"""
	NSFW Predictor based on LAION dataset
	Colab: https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

	"""

	def __init__(self, cfg):

		self.device = cfg['device']
		self.nsfw_model = self.get_nsfw_model(clip_model=cfg['model'], cache_folder=cfg['cache_folder'])
		self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
		self.model = self.model.to(self.device)


	def get_nsfw_model(self, clip_model="vit_l_14", cache_folder='./'):
		loaded_model = load_model(cache_folder, custom_objects=ak.CUSTOM_OBJECTS)
		loaded_model.predict(np.random.rand(10**3, 768).astype("float32"), batch_size=10**3)
		return loaded_model

	def normalized(self, a, axis=-1, order=2):
		l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
		l2[l2 == 0] = 1
		return a / np.expand_dims(l2, axis)

	def calculate(self, data):
		image_ = Image.open(data['image']['path'])
		image_ = self.preprocess(image_).unsqueeze(0).to(self.device)
		with torch.no_grad():
			image_features = self.model.encode_image(image_)
		emb = np.asarray(self.normalized(image_features.detach().cpu()))
		prediction = self.nsfw_model.predict(emb)
		return {'nsfw': float(prediction[0][0])}

	def verify(self, img_path):
		import time
		start_time = time.time()
		image_ = Image.open(img_path)
		image_ = self.preprocess(image_).unsqueeze(0).to(self.device)
		with torch.no_grad():
			image_features = self.model.encode_image(image_)
		emb = np.asarray(self.normalized(image_features.detach().cpu()))
		prediction = self.nsfw_model.predict(emb)
		print('Time cost: ', time.time() - start_time)
		return {'nsfw': prediction[0][0]}


if __name__ == "__main__":
	# In the terminal, using the folloing command:
	# >>> python evaluator/nfsw_metric.py
	cfg = {
		'model': "vit_l_14",
    	'cache_folder': "/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/pretrain_model/clip_autokeras_binary_nsfw",
		'device': 'cuda:0'
	}
	metric = NSFWMetric(cfg)
	print(metric.verify('/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/data/images/0a1a2d23-6e76-45ad-8177-aff945161575.jpg'))
