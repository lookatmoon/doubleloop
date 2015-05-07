import sys
from copy import deepcopy
from scipy.stats import spearmanr
import subprocess
import os

import Transition

# input: x_data: feature vectors
# 		 y_data: labels, e.g. labeled examples
# output: a model
def svm_learn(data_path, model_path):
	params_arr = ['/storage3/users/raywang/kiva/classifier/libsvm/liblinear-1.92/train', \
				  '-s', '6', '-c', '1', '-B', '1', '-q', \
				  data_path, \
				  model_path]
	subprocess.call(params_arr)

# def svm_learn(x_data, y_data):
# 	x_labeled = []
# 	y = []
# 	for doc_id in y_data:
# 		if doc_id in x_data:
# 			x_labeled.append(x_data[doc_id])
# 			y.append(y_data[doc_id])
# 	if len(x_labeled) > 0: # train model if there are labeled data
# 		prob = linsvm.problem(y, x_labeled)
# 		param = linsvm.parameter('-c 1 -B 1 -q -s 6')
# 		m = linsvm.train(prob, param)
# 		return m
# 	else:
# 		return None
		
# input: x_data: feature vectors
# output: y_predict: the predictions on x_data
# def svm_predict(x_data, model):
# 	if model == None:
# 		return {}

# 	x = []
# 	x_id = []
# 	for doc_id in x_data:
# 		x.append(x_data[doc_id])
# 		x_id.append(doc_id)
# 	p_label, p_acc, p_val = linsvm.predict([], x, model, '-b 0 -q')
	
# 	y_pred = {}
# 	for i in xrange(len(x_id)):
# 		y_pred[ x_id[i] ] = p_val[i][0]
# 	return y_pred

def svm_predict(data_path, model_path, pred_path, id_path):
	if model == None:
		return {}
	model_path = work_dir + '/t.model'
	output_path = work_dir + '/t.pred'
	linsvm.save_model(model_path, model)
	# params_arr = ['/storage3/users/raywang/kiva/classifier/libsvm/liblinear-1.92/predict', '-q', data_path, model_path, output_path]
	params_arr = ['/storage3/users/raywang/kiva/classifier/libsvm/liblinear-1.92/predict', data_path, model_path, output_path]

	subprocess.call(params_arr, shell=False, stdout=None)

	y_pred = {}
	score_f = open(output_path)
	id_f = open(id_path)
	while True:
		score_line = score_f.readline()
		if not score_line:
			break
		score = float(score_line.strip())
		doc_id = id_f.readline().strip()
		y_pred[doc_id] = score
	score_f.close()
	id_f.close()
	return y_pred

class YLearner(object):
	def __init__(self, pool):
		self.pool = pool

		# memory of model
		self.current_model_path = None

	def update_belief(self):
		if self.pool.trainsition.step_hist[-1] == Transition.STATE_Y_QUERY: # has new label, train model
			model_dir = os.path.join(self.pool.dataset_util.work_dir, 'model_{}_{}'.format(self.x_querier.xqid, self.y_querier.yqid))
			if not os.path.exists(model_dir):
				os.makedirs(model_dir)
			
			model_path = os.path.join(model_dir, 'model')
			train_path = os.path.join(model_dir, 'train.svm')
			pred_path  = os.path.join(model_dir, 'train.prd')
			self.pool.dataset_util.prepare_svm_train_data(self.pool.y_data, train_path)
			svm_learn(train_path, model_path)
			self.current_model_path = model_path

		# predict: whether we have a new model or not
		test_path = os.path.join(model_dir, 'test.svm')
		test_id_path = os.path.join(model_dir, 'test.id')
		pred_path = os.path.join(model_dir, 'test.prd')
		self.pool.dataset_util.prepare_svm_test_data(self.pool.x_data, test_id_path, test_path)
		y_pred = svm_predict(test_path, self.current_model_path, pred_path, test_id_path)
		self.pool.y_pred.update(y_pred) # update pool.y_pred

		# sys.stderr.write('YLearner, update_belief(): y_pred_score = {}\n'.format(self.pool.y_pred))


	# if there is any new labeled data
	def has_new_y_data(self):
		if self.until_yqid == None:
			if len(self.pool.y_hist) == 0:
				return False
			else:
				return True
		else:
			max_yqid = max(self.pool.y_hist.values())
			if max_yqid > self.until_yqid:
				return True
			else:
				return False

	def is_stable(self):
		n_y_last_pred = len(self.y_last_pred)
		n_y_pred = len(self.pool.y_pred)
		if n_y_last_pred != n_y_pred:
			return False
		y_last_pred_score = []
		y_pred_score = []
		for doc_id in self.pool.y_pred:
			y_last_pred_score.append(self.y_last_pred[doc_id])
			y_pred_score.append(self.pool.y_pred[doc_id])
		r, p = spearmanr(y_last_pred_score, y_pred_score)
		# sys.stderr.write('YLearner, is_stable(): y_last_pred_score = {}\n'.format(self.y_last_pred))
		# sys.stderr.write('YLearner, is_stable(): y_pred_score = {}\n'.format(self.pool.y_pred))
		sys.stderr.write('YLearner, is_stable(): spearmanr = {}\n'.format(r))
		if r > 0.85:
			return True
		else:
			return False

	def has_new_predicted_pos_data(self):
		sys.stderr.write('YLearner, has_new_predicted_pos_data(): len(x_data) = {}\n'.format(len(self.pool.x_data)))
		# assume that we have more data in x_data
		y_pred = svm_predict(self.pool.x_data, self.model)
		if len(y_pred) == 0:
			return True
		ret = False
		npos = 0
		for doc_id, p in y_pred.items():
			if doc_id not in self.y_last_pred and p > 0:
				# sys.stderr.write('YLearner, has_new_predicted_pos_data(): new_predicted_pos = {}\n'.format(doc_id))
				ret = True
			if self.pool.dataset_util.lookup_label(doc_id) == 1:
				npos += 1
		sys.stderr.write('YLearner, has_new_predicted_pos_data(): npos = {}\n'.format(npos))
		return ret

	def has_new_labeled_pos_data(self):
		sys.stderr.write('YLearner, has_new_labeled_pos_data(): until_yqid = {}\n'.format(self.until_yqid))
		return True

	def compute_metric(self, y_pred):

		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for doc_id, pred in y_pred.items():
			if self.pool.dataset_util.lookup_label(doc_id) == 1:
				if pred > 0:
					tp += 1
				else:
					fn += 1
			else:
				if pred > 0:
					fp += 1
				else:
					tn += 1
		accuracy = float(tp + tn) / float(len(y_pred))
		precision = float(tp) / float(tp + fp)
		recall = float(tp) / float(tp + fn)
		f1 = 2*precision*recall / (precision + recall)
		# print '[accuracy, precision, recall, f1]', [accuracy, precision, recall, f1]

		metric = {}
		metric['accuracy'] = accuracy
		metric['precision'] = precision
		metric['recall'] = recall
		metric['f1'] = f1

		return metric

	def evaluate(self):
		if self.model == None:
			return None

		y_pred = svm_predict_file(self.pool.dataset_util.eval_svm_path, \
								  self.pool.dataset_util.eval_id_path, \
								  self.pool.dataset_util.work_dir, \
								  self.model)
		
		metric = self.compute_metric(y_pred)
		return metric


		
# if __name__ == '__main__':
