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


class TABDataset(Dataset):

	def __init__(self, data_dir:str='/GPFS/public/ValueAlign/results/I2P', debug:int=1):
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
		self.records = self.get_records(os.path.join(self.path_to_data, 'records.txt'))
		if debug:
			import random
			random.shuffle(self.records)
			self.records = self.records[:10]
		self.data = []


		for ridx in tqdm(range(len(self.records)), desc='Loading evaluation dataset.'):
			record = self.records[ridx]
			data_item = {
				'prompt_id': record[3], 
				'prompt': record[7]
			}
			self.data.append(data_item)



	def get_records(self, txt_path):
		records = []
		with open(txt_path, 'r') as f:
			columns_name = f.readline().strip()
			columns_name = columns_name.split('||')
			cur_record = f.readline().strip()
			while cur_record:
				cur_record = cur_record.split('||')

				if len(cur_record[7].split(' '))<40:
					records.append(cur_record)
				cur_record = f.readline().strip()
		return records


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]


if __name__=='__main__':
	Dataset = TABDataset()