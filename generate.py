import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from dataclasses import replace
from typing import List, Optional
from tqdm import tqdm
import threading
import json
import time

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  
config.gpu_options.allow_growth = True	  
sess = tf.compat.v1.Session(config = config)

from common.config import Config, print_log, prepare_seed


def build_generator(cfg, args):
	'''
	1. Return the prompt dataloader
	2. Construct the generators (the whole pipeline)
	'''

	# construct the prompt dataloader
	if cfg.dataset == 'I2P':
		from dataloader.I2P_dataloader import I2PDataset
		dataloader = I2PDataset(data_dir=cfg.dataset_path, debug=args.debug)
	elif cfg.dataset == 'TAB':
		from dataloader.TAB_dataloader import TABDataset
		dataloader = TABDataset(data_dir=cfg.dataset_path, debug=args.debug)
	elif cfg.dataset == 'TAB100':
		from dataloader.TAB100_dataloader import TAB100Dataset
		reproduce = False
		try: 
			reproduce=cfg.reproduce
		except: 
			pass
		dataloader = TAB100Dataset(data_dir=cfg.dataset_path, dataset=cfg.dataset, debug=args.debug, reproduce=reproduce)
	elif cfg.dataset == 'MSCOCO': # serving as the safe/easy dataset!
		from dataloader.MSCOCO_dataloader import MSCOCODataset
		dataloader = MSCOCODataset(data_dir=cfg.dataset_path, stage=cfg.stage, length=cfg.length)
	elif cfg.dataset == 'TOX1K': # serving as the hard dataset
		from dataloader.TAB100_dataloader import TAB100Dataset
		reproduce = False
		try: reproduce=cfg.reproduce
		except: pass
		dataloader = TAB100Dataset(data_dir=cfg.dataset_path, dataset=cfg.dataset, debug=args.debug, reproduce=reproduce)
	elif cfg.dataset == 'TAB40':
		from dataloader.TAB40_dataloader import TAB40Dataset
		dataloader = TAB40Dataset(data_dir=cfg.dataset_path, debug=args.debug)
	elif cfg.dataset == 'TAB100_toxic':
		from dataloader.TAB100_dataloader import TAB100Dataset
		reproduce = False
		dataloader = TAB100Dataset(data_dir=cfg.dataset_path, dataset=cfg.dataset, debug=args.debug, reproduce=reproduce)
	else:
		print('Dataset {} has not been implemented!'.format(cfg.dataset))
		raise ValueError


	generator = {}
	for gen_name in cfg.generators.keys():
		if gen_name=='baseline':
			from generator.baseline import Baseline
			generator.update({gen_name: Baseline(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		elif gen_name=='baseline_safe':
			from generator.baseline import Baseline_Safe
			generator.update({gen_name: Baseline_Safe(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		elif gen_name=='sc_yi':
			from generator.llm_sc import sc_yi
			generator.update({gen_name: sc_yi(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		elif gen_name=='sc_qwen':
			from generator.llm_sc import sc_qwen
			generator.update({gen_name: sc_qwen(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		elif gen_name=='sc_gpt':
			from generator.llm_sc import sc_gpt
			generator.update({gen_name: sc_gpt(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		elif gen_name=='dalle3':
			from generator.dalle3_generator import DallE3Generator
			generator.update({gen_name: DallE3Generator(cfg.generators[gen_name], image_save_path=cfg.image_dir)})
		else:
			print('Generator {} has not been implemented!'.format(gen_name))
			raise ValueError

	return dataloader, generator

def generate(prompt_dataset, generator, conf_generator):
	print('Sequences in the prompt dataset: ', len(prompt_dataset))
	for cnt in tqdm(range(len(prompt_dataset)), desc='Generating samples'):
		# generate images from prompt_dataset
		data = prompt_dataset[cnt]
		for gen_key in generator.keys():
			data.update(generator[gen_key].inference(data, cnt))
		write_into_json(conf_generator, data)
		
	return prompt_dataset

def write_into_json(conf_generator, data):
	# print('Start to write into json files')
	if  data['status']=='SUCCESS':
		del data['image']['data'], data['image']['data_torch']
	json_path = '{}/records_json.json'.format(conf_generator.log_dir)
	# import pdb; pdb.set_trace()
	with open(json_path, 'a', encoding='utf-8') as file:
		json.dump(data, file, ensure_ascii = False)
		file.write('\n')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-cg",
		"--conf-generator",
		type=str, 
		default='', 
		help='Path to the config file of the T2I generator'
	)

	parser.add_argument(
		"-i",
		"--info", 
		type=str, 
		default='debug', 
		help='Information to identify this run',
	)

	parser.add_argument(
		"-d",
		"--debug", 
		type=int, 
		default=1, 
		help='Whether in debug mode or not',
	)

	args = parser.parse_args()

	conf_generator = Config(args.conf_generator, args.info, create_dirs=True)
	prompt_dataset, generator = build_generator(conf_generator, args)

	generate(prompt_dataset, generator, conf_generator)
	print('ALL MISSION FINISHED!')

if __name__ == "__main__":
	main()

