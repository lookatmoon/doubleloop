import os.path
import json, sys
import subprocess

from x_oracle import XOracle
from dataset_utility import DataSetUtility

def svm_predict(data_path, model_path, pred_path, id_path):

	params_arr = ['/storage3/users/raywang/kiva/classifier/libsvm/liblinear-1.92/predict', \
					'-b', '1', \
					data_path, \
					model_path, \
					pred_path]
	subprocess.call(params_arr, shell=False, stdout=None)

def get_id_prediction(pred_path, id_path):
		pred = {}
		score_f = open(pred_path)
		id_f = open(id_path)
		ss = score_f.readline().strip().split()
		pos = ss.index('1')
		while True:
			line = score_f.readline()
			if not line:
				break
			score = float(line.strip().split()[pos])
			doc_id = id_f.readline().strip()
			pred[doc_id] = score
		score_f.close()
		id_f.close()
		return pred

if __name__ == '__main__':

	'''
	# dummy model
	m = {}
	m['query_arr'] = []
	q = {}
	q['q'] = {'currenc':3.5, 'union':2.92, 'european':4.48, 'market':3.78}
	q['r'] = 500
	m['query_arr'].append(q)

	q = {}
	q['q'] = {'currenc':3.91, 'union':2.877, 'european':4.27, 'emu':2.19}
	q['r'] = 500
	m['query_arr'].append(q)

	m['classifier'] = '/storage6/foreseer/users/raywang/pool/al/G151/RR/9.model'

	f = open('model.json', 'w')
	json.dump(m, f, sort_keys=True, indent=2)
	f.close()
	'''

	########################################################################
	if len(sys.argv) != 4:
		exit ('Params: model_dir json_name data_id_path')

	model_dir = sys.argv[1]
	json_name = sys.argv[2]
	data_id_path = sys.argv[3]
	out_dir = model_dir

	model_json_path = os.path.join(model_dir, 'model.json')
	
	f = open(model_json_path)
	m = json.load(f)
	f.close()

	# prepare utilities
	util = DataSetUtility(os.path.abspath(data_id_path))

	# union of queries => x_data
	x_data = {}
	for q in m['query_arr']:
		# query_str = ' '.join('{}^{}'.format(term, weight) for term, weight in q['q'].items())
		query_str = q['q']
		params = {'max':q['r'], 'query': query_str}
		dat = util.search_index(params)
		dat = util.restrict_id(dat)
		vec = util.parse_data_to_sparse_vec(dat)
		for doc_id in vec:
			if doc_id not in x_data:
				x_data[doc_id] = vec[doc_id]

	# apply the classifier
	id_path = os.path.join(out_dir, 'pool.id')
	svm_path = os.path.join(out_dir, 'pool.svm')
	util.prepare_svm_test_data(x_data, id_path, svm_path)

	model_path = m['classifier']
	pred_path = os.path.join(out_dir, 'pool.prd')
	svm_predict(svm_path, model_path, pred_path, id_path)
	pred = get_id_prediction(pred_path, id_path)

	metric = util.compute_metrics(pred)

	# write results out
	res_path = os.path.join(out_dir, 'result.txt')
	f = open(res_path, 'w')
	for doc_id in pred:
		f.write('{}\t{}\n'.format(doc_id, pred[doc_id]))
	f.close()

	result_json_path = os.path.join(out_dir, json_name + '.json')
	f = open(result_json_path, 'w')
	json.dump(metric, f, sort_keys=True, indent=2)
	f.close()



	
