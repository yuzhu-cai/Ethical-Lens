import torch
from torchmetrics.multimodal.clip_score import CLIPScore

class ObjectDetectionMetric():
	"""
	
	"""

	def __init__(self, cfg, multilingual: bool = False):
		self._multilingual: bool = multilingual
		self.metric = CLIPScore(model_name_or_path=cfg['model'])

	def calculate(self, data):
		score = self.metric(data['image']['data_torch'], data['prompt'])
		return score.detach().item()
