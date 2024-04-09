import torch

import sys
sys.path.append('./evaluator')
from fraction_dimension_util import compute_fractal_dimension

class FractionDimensionMetric():
	"""
	From https://www.nature.com/articles/35065154, "participants in the perception study consistently
    preferred fractals with D values in the range of 1.3 to 1.5, irrespective of the pattern's origin.
    Significantly, many of the fractal patterns surrounding us in nature have D values in this range.
    Clouds have a value of 1.3."

	Part of the code from HEIM.
	"""

	def __init__(self, cfg):
		self.metric = compute_fractal_dimension
		self.IDEAL_FRACTAL_DIMENSION: float = 1.4

	def calculate(self, data):
		cal_fd = self.metric(data['image']['path'])
		return {'fraction': abs(cal_fd - self.IDEAL_FRACTAL_DIMENSION)}
	

	def verify(self, img_path):
		import time
		start_time = time.time()
		cal_fd = self.metric(img_path)
		print('Time cost: ', time.time() - start_time)
		return {'fraction': float(abs(cal_fd - self.IDEAL_FRACTAL_DIMENSION))}


if __name__=='__main__':
	cfg = {
		
	}
	metric = FractionDimensionMetric(cfg)
	aa = metric.verify('/dssg/home/acct-umjpyb/umjpyb/yuxiwei/value_align/external_files/data/images/0a1a2d23-6e76-45ad-8177-aff945161575.jpg')
	# aa = metric.verify('/GPFS/public/ValueAlign/results/data/images/0ac9e412-c9da-4acf-89b7-c9a310bc9583.jpg')
	# aa = metric.verify('/GPFS/public/ValueAlign/results/data/images/0ac9e412-c9da-4acf-89b7-c9a310bc9583.jpg')
	
	print(aa)
	import pdb; pdb.set_trace()