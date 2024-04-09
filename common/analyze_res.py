import os
import pandas as pd


def  get_records_from_txt(filename):

	records = []
	with open(filename, 'r') as f:
		columns_name = f.readline().strip()
		columns_name = columns_name.split('||')
		cur_record, count = f.readline().strip(), 0
		while cur_record and count<900:
			cur_record = cur_record.split('||')
			cur_record[-3] = (0 if cur_record[-3]=='False' else 1)
			cur_record[-1] = (0 if cur_record[-1]=='nan' else cur_record[-1])
			for i in [4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18]:
				try:
					cur_record[i] = eval(cur_record[i])
				except:
					pass
			records.append(cur_record)
			cur_record = f.readline().strip()
			count += 1

	df = pd.DataFrame(records, columns=columns_name)
	return df

results_path = "/DB/rhome/weibomao/GPFS/value_align/results/tab100_dalle3_try1/log/records_gen.txt"
# results_path = "/DB/rhome/weibomao/GPFS/value_align/results/tab100_baseline_try1/log/records_gen.txt"
# results_path = '/DB/rhome/weibomao/GPFS/value_align/results/tab_try3/log/records_gen.txt'
df = get_records_from_txt(results_path)
df_success = df[df['status']=='SUCCESS']
print(df_success.describe())
print(1-len(df_success)/len(df))
