import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from PIL import Image
from dataclasses import replace
from typing import List, Optional
from torchvision import transforms
from tqdm import tqdm
import threading
import json
import time
import numpy as np

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)

from common.config import Config, print_log, prepare_seed


def read_json_files(json_path):
	json_data = []
	with open(json_path, 'r') as file:
		for line in file:
			json_data.append(json.loads(line))
	return json_data

def build_evaluator(cfg, args):
	'''
	
	'''
	evaluators = {}
	# the following code could be optimized
	for eval_name in cfg.evaluators.keys():
		# import pdb; pdb.set_trace()
		if eval_name=='clip_score':
			from evaluator.clip_score_metric import CLIPScoreMetric
			evaluators.update({eval_name: CLIPScoreMetric(cfg.evaluators[eval_name])})
		elif eval_name=='nude':
			from evaluator.nude_metric import NudeMetric
			evaluators.update({eval_name: NudeMetric(cfg.evaluators[eval_name])})
		elif eval_name=='aesthetic':
			from evaluator.aesthetic_metric import AestheticMetric
			evaluators.update({eval_name: AestheticMetric(cfg.evaluators[eval_name])})
		elif eval_name=='nsfw':
			from evaluator.nsfw_metric import NSFWMetric
			evaluators.update({eval_name: NSFWMetric(cfg.evaluators[eval_name])})
		elif eval_name=='watermark':
			from evaluator.watermark_metric import WatermarkMetric
			evaluators.update({eval_name: WatermarkMetric(cfg.evaluators[eval_name])})
		elif eval_name=='fraction':
			from evaluator.fraction_dimension_metric import FractionDimensionMetric
			evaluators.update({eval_name: FractionDimensionMetric(cfg.evaluators[eval_name])})
		elif eval_name=='llava_toxic':
			from evaluator.llava_toxicity_metric import LLaVAToxicMetric
			evaluators.update({eval_name: LLaVAToxicMetric(cfg.evaluators[eval_name])})
		elif eval_name=='llava_gender':
			from evaluator.llava_bias_metric import LLaVABiasMetric
			evaluators.update({eval_name: LLaVABiasMetric(cfg.evaluators[eval_name])})
		elif eval_name=='gpt4v_toxic':
			from evaluator.gpt4v_toxicity_metric import GPT4VToxicMetric
			evaluators.update({eval_name: GPT4VToxicMetric(cfg.evaluators[eval_name])})
		elif eval_name=='gpt4v_bias':
			from evaluator.gpt4v_bias_metric import GPT4VBiasMetric
			evaluators.update({eval_name: GPT4VBiasMetric(cfg.evaluators[eval_name])})

	return evaluators

def single_evaluator_thread(calculation_func, data, eval_key):
	# start_time = time.time()
	data['evaluation_results'].update(calculation_func(data))
	# print('{} cost: {}'.format(eval_key, time.time() - start_time))

def evaluate(generated_dataset, evaluators, conf_evaluator):
	print('Sequences in the generated dataset: ', len(generated_dataset))
	image_transform = transforms.ToTensor()
	for cnt in tqdm(range(len(generated_dataset)), desc='Evaluatiing samples'):
		# evaluate images from generated_dataset
		data = generated_dataset[cnt]
		data['evaluation_results'] = {}
		if data['status']=='SUCCESS':
			img_path = data["image"]["path"]
			raw_image = Image.open(img_path).resize((256, 256))
			raw_image_np = np.array(raw_image)
			data['image']['data'] = raw_image_np
			data['image']['data_torch'] = (image_transform(raw_image_np)*255).byte()
			# Multi-threads evaluation
			tmp_threads = []
			for eval_key in evaluators.keys():
				thread = threading.Thread(target=single_evaluator_thread, args=(evaluators[eval_key].calculate, data, eval_key))
				tmp_threads.append(thread)
				thread.start()
				# data['evaluation_results'].update(evaluators[eval_key].calculate(data))
			# Wait for all threads to complete
			for thread in tmp_threads:
				thread.join()
		
		write_into_json(conf_evaluator, data)
	
	return

def write_into_json(conf_generator, data):
	# print('Start to write into json files')
	if  data['status']=='SUCCESS':
		del data['image']['data'], data['image']['data_torch']
	json_path = '{}/records_json_.json'.format(conf_generator.log_dir)
	# import pdb; pdb.set_trace()
	with open(json_path, 'a', encoding='utf-8') as file:
		json.dump(data, file, ensure_ascii = False)
		file.write('\n')
	# import pdb; pdb.set_trace()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-ce",
		"--conf-evaluator", 
		required=True,
		type=str, 
		default='', 
		help='Path to the config file of the evaluator',
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

	conf_evaluator = Config(args.conf_evaluator, args.info, create_dirs=True)
	evaluators = build_evaluator(conf_evaluator, args)

	json_path = '{}/records_json.json'.format(conf_evaluator.log_dir)
	generated_dataset = read_json_files(json_path)

	evaluate(generated_dataset, evaluators, conf_evaluator)
	print('ALL MISSION FINISHED!')

if __name__ == "__main__":
	main()