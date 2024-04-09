from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from numpy import asarray
import torch
from tqdm import tqdm
import os
import pandas as pd


def collate_batch(batch):
	
	return batch


class I2PDataset(Dataset):

	def __init__(self, data_dir:str='/GPFS/public/ValueAlign/results/I2P', debug:int=1):
		"""
		Parameters
		----------
		data : {
			'prompt_id': 0001,
			'prompt': str, 
			'categories', 
			'hard', 
			'inappropriate_percentage', 
			'nudity_percentage', 
			'q16_percentage', 
			'sd_safety_percentage', 
			'prompt_toxicity'
		}

		Returns
		----------
		"""
		self.path_to_data = data_dir
		self.records = self.get_records(os.path.join(self.path_to_data, 'i2p_benchmark.csv'))
		if debug:
			self.records = self.records[:10]
		self.data = []

		# reserved_keys = ['prompt', 'categories', 'hard', 'inappropriate_percentage', 'nudity_percentage', \
		# 		'q16_percentage', 'sd_safety_percentage', 'prompt_toxicity']
		reserved_keys = ['prompt', 'categories']

		for ridx in tqdm(range(len(self.records)), desc='Loading evaluation dataset.'):
			record = self.records.loc[ridx]
			
			data_item = {
				'prompt_id': '{:0>6d}'.format(ridx), 
			}
			data_item.update({rk: record[rk] for rk in reserved_keys})
			self.data.append(data_item)


	def get_records(self, csv_path):
		data = pd.read_csv(csv_path, sep=',', header=0)
		return data


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]


if __name__=='__main__':
	Dataset = I2PDataset()