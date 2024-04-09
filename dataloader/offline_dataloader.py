from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from numpy import asarray
import torch
from tqdm import tqdm
import os

def collate_batch(batch):
	
	return batch


class LexicaDataset(Dataset):

	def __init__(self, data_dir:str='/GPFS/public/ValueAlign/results/data', debug:int=1):
		"""
		Parameters
		----------
		data : {
			'prompt': str,
			'status': 'SUCCESS'/'FAIL',
			'time': float (ms),
			'image': {
				'path': str,
				'data': np.array (3, H, W),
			}
		}

		Returns
		----------
		"""
		self.path_to_data = data_dir
		self.records = self.get_records(os.path.join(self.path_to_data, 'records.txt'))
		if debug:
			self.records = self.records[1:5]
		self.data = []

		for ridx in tqdm(range(len(self.records)), desc='Loading evaluation dataset.'):
			record = self.records[ridx]
			path_to_image = '{}/images/{}.jpg'.format(self.path_to_data, record[3])
			img_data = Image.open(path_to_image)
			data_item = {
				'id': record[3], 
				'prompt': record[7],
				'status': 'SUCCESS',
				'time': 0,
				'image': {
					'path': path_to_image,
					'data': asarray(img_data),
					'data_torch': torch.from_numpy(np.transpose(asarray(img_data),  (2,0,1)).copy()),
					'width': record[8],
					'height': record[9]
				}
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
				records.append(cur_record)
				cur_record = f.readline().strip()
		return records


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		return self.data[idx]

