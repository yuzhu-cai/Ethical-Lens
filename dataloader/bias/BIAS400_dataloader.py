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


class BIAS400Dataset(Dataset):

	def __init__(self, data_dir:str='/GPFS/public/ValueAlign/results/data', debug:int=1):
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
		self.data = self.get_records(os.path.join(self.path_to_data, 'BIAS400_records.json'))
		if debug:
			import random
			random.shuffle(self.data)
			self.data = self.data[:10]
		

	def get_records(self, json_path):
		with open(json_path, 'r') as file:
			json_data = json.load(file)
		return json_data


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]


if __name__=='__main__':
	Dataset = BIAS400Dataset()