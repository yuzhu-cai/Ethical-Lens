from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from numpy import asarray
import torch
from tqdm import tqdm
import os
import pandas as pd
import json

def collate_batch(batch):
	
	return batch


class TAB100Dataset(Dataset):

	def __init__(self, data_dir:str='/GPFS/public/ValueAlign/results/I2P', dataset:str='TAB100', debug:int=1, reproduce:bool=False):
		"""
		Parameters
		----------
		data : {
			'prompt_id': 0001,
			'prompt': str, 
		}

		Returns
		----------
		"""
		self.path_to_data = data_dir
		if reproduce:
			self.data = self.get_records_reproduce(os.path.join(self.path_to_data, 'records_json.json'))
		else:
			self.data = self.get_records(os.path.join(self.path_to_data, 'records_{}.json'.format(dataset[3:])))
		if debug:
			import random
			random.shuffle(self.data)
			self.data = self.data[:10]
		

	def get_records(self, json_path):
		with open(json_path, 'r') as file:
			json_data = json.load(file)
		return json_data

	def get_records_reproduce(self, json_path):
		json_data = []
		reserve_key = ['prompt_id', 'prompt', 'aspect', 'type', 'keyword', 'gallery', 'src', 'model', 'seed', 'guidance', 'prompt_chinese', 'random_seed']
		with open(json_path, 'r') as file:
			for line in file:
				line = json.loads(line)
				tmp = {k: line[k] for k in reserve_key}
				json_data.append(tmp)
		return json_data


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]


if __name__=='__main__':
	Dataset = TAB100Dataset()