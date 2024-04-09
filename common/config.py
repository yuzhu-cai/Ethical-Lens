import yaml
import os
import os.path as osp
import shutil
import glob
import numpy as np
import random
import torch
from easydict import EasyDict



def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)


def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file

	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	if same_line: log.write('{}'.format(print_str))
	else: log.write('{}\n'.format(print_str))
	log.flush()

    
def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


class Config:
    def __init__(self, cfg_id, info='debug', create_dirs=False):
        self.name = cfg_id
        # print(cfg_path)
        files = glob.glob(cfg_id, recursive=True)
        assert len(files) == 1, 'There are zero/multiple config files named: {}'.format(cfg_id)
        self.yml_dict = EasyDict(yaml.safe_load(open(files[0], 'r')))
        if create_dirs:
            self.results_root_dir = os.path.expanduser(self.yml_dict['results_root_dir'])

            self.cfg_dir = '%s/%s' % (self.results_root_dir, info)
            self.result_dir = '%s/results' % self.cfg_dir
            self.log_dir = '%s/log' % self.cfg_dir
            self.file_dir = '%s/files' % self.cfg_dir
            self.image_dir = '%s/generated_images' % self.cfg_dir
            os.makedirs(self.result_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.file_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
        
        
        
    def __getattribute__(self, name):
        yml_dict = super().__getattribute__('yml_dict')
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__('yml_dict')
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
            