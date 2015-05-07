import sys, os, random
import subprocess
from copy import deepcopy
from sklearn import metrics
import operator

def compute_rprec(y_arr, s_arr):
	sort_by_score = sorted(zip(y_arr, s_arr), key = operator.itemgetter(1), reverse = True)
	r = sum(y_arr)
	s = 0
	for i in xrange(len(sort_by_score)):
		s += sort_by_score[i][0]
		if s == r:
			return float(r) / float(i + 1)

class DataSetUtility(object):
	def __init__(self):
		# x-data
		self.raw_dir = '/storage6/foreseer/users/raywang/large_learn_data/rcv1-v2/text/processed'
		self.search_cmd = ['java', '-jar', '/storage6/foreseer/users/raywang/pool/index/rcv/search.jar']
		self.index_path = '/storage6/foreseer/users/raywang/pool/index/rcv/index_rcv'
		self.term_idf_path = '/storage6/foreseer/users/raywang/large_learn_data/rcv1-v2/text/stem.termid.idf.map.txt'
		self.default_search_params = {'index':self.index_path, 'out':'docID;text'}

		# x,y data
		self.train_data_path = '/storage6/foreseer/users/raywang/pool/al/train.dat'

		# y-data
		# self.label_path = '/storage6/foreseer/users/raywang/large_learn_data/rcv1-v2/label/industries/rcv1-v2.label'
		self.label_path = '/storage6/foreseer/users/raywang/large_learn_data/rcv1-v2/label/topics/rcv1-v2.label'

		# task specific
		self.positive_labels = {'G151':1}
		self.work_dir = '/storage6/foreseer/users/raywang/pool/dl/dat/G151'
		self.x_seed_path = '/storage6/foreseer/users/raywang/pool/dl/dat/G151/x_seed.txt'
		self.id_set_path = '/storage6/foreseer/users/raywang/pool/dl/dat/G151/train.id'
		# self.id_set_path = '/storage6/foreseer/users/raywang/pool/dl/dat/G151/test.id'

		# load term map: id should be integers starting from 1
		self.term2id = {}
		self.id2term = {}
		f = open(self.term_idf_path)
		for line in f:
			t, tid, idf = line.strip().split()
			tid = int(tid)
			self.term2id[t] = tid
			self.id2term[tid] = t
		f.close()

		# load labels
		self.label = {}
		f = open(self.label_path)
		for line in f:
			doc_id, label = line.strip().split()
			if doc_id not in self.label:
				if label in self.positive_labels:
					self.label[doc_id] = 1
				else:
					self.label[doc_id] = -1
			else: # if old = +1: pass. if old=-1,new=+1: overwrite. if old=-1,new=-1: pass
				if self.label[doc_id] == -1 and label in self.positive_labels:
					self.label[doc_id] = 1
		f.close()

		# prepare for work dir
		if not os.path.exists(self.work_dir):
			os.makedirs(self.work_dir)

		# load id set
		self.id_set = {}
		f = open(self.id_set_path)
		for line in f:
			doc_id = line.strip()
			self.id_set[doc_id] = None
		f.close()

		self.num_pos = sum([1 for doc_id, label in self.label.items() if doc_id in self.id_set and label == 1])
		self.num_neg = sum([1 for doc_id, label in self.label.items() if doc_id in self.id_set and label == -1])

	# dat[doc_id] = 'term1 term2 term 2 term3'
	# vec[doc_id][term_id] = count # could be weighted by IDF...
	def parse_data_to_sparse_vec(self, dat):
		vec = {}
		for doc_id, text in dat.items():
			vec[doc_id] = {}
			for term in text.split():
				if term in self.term2id:
					tid = self.term2id[term]
					if tid not in vec[doc_id]:
						vec[doc_id][tid] = 1
					else:
						vec[doc_id][tid] += 1
		return vec

	# vec[term_id] = weight
	def parse_sparse_vec_to_query_str(self, vec):
		q = []
		for tid in vec:
			if tid in self.id2term:
				term = self.id2term[tid]
				q.append('{}^{}'.format(term, vec[tid]))
		q_str = ' '.join(q)
		print q_str
		return q_str

	def search_index(self, params):
		search_par = deepcopy(self.default_search_params)
		search_par.update(params)
		params_arr = deepcopy(self.search_cmd)
		for k, v in search_par.items():
			params_arr.append('-' + k)
			params_arr.append(str(v))
		# print params_arr
		searcher = subprocess.Popen(params_arr, shell=False, stdout=subprocess.PIPE)
		dat = {}
		while True:
			line = searcher.stdout.readline()
			if not line:
				break
			dat_id, text = line.strip().split('\t')
			dat[dat_id] = text
		return dat

	def lookup_label(self, doc_id):
		if doc_id in self.label:
			return self.label[doc_id]
		sys.stderr.write('DataSetUtility, lookup_label(): unknown doc_id: {}\n'.format(doc_id))
		return -1

	def random_sample_docs_by_label(self, label, num_doc):
		arr = [i for i in self.label if self.label[i] == label]
		return random.sample(arr, num_doc)

	def random_sample_docs_by_label_no_eval(self, label, num_doc):
		arr = [i for i in self.label if self.label[i] == label and i not in self.eval_id]
		return random.sample(arr, num_doc)

	def scan_raw_by_doc_id(self, doc_ids):
		dic = {}
		for doc_id in doc_ids:
			dic[doc_id] = 1
		dat = {}
		for filepath in [os.path.join(root, file) for root, dirs, files in os.walk(self.raw_dir) for file in files]:
			sys.stderr.write('DataSetUtility, scan_raw_by_doc_id(): scanning:\n{}\n'.format(filepath))
			f = open(filepath)
			for line in f:
				ss = line.strip().split('\t')
				if len(ss) == 2:
					if ss[0] in dic:
						dat[ss[0]] = ss[1]
			f.close()
		return dat

	def write_dat_to_file(self, dat, path):
		f = open(path, 'w')
		for doc_id in dat:
			f.write('{}\t{}\n'.format(doc_id, dat[doc_id]))
		f.close()

	def read_dat_from_file(self, path):
		dat = {}
		f = open(path)
		for line in f:
			ss = line.strip().split('\t')
			if len(ss) == 2:
				dat[ss[0]] = ss[1]
		f.close()
		return dat

	def write_vec_label_to_file(self, vec, path):
		f = open(path, 'w')
		for doc_id, v in vec.items():
			label = self.lookup_label(doc_id)
			f.write('{}\t{}'.format(doc_id, label))
			for fid in sorted(v.keys()):
				f.write(' {}:{}'.format(fid, v[fid]))
			f.write('\n')
		f.close()

	def restrict_id(self, dic):
		intersection = []
		for k in dic:
			if k not in self.id_set:
				intersection.append(k)
		for k in intersection:
			dic.pop(k, None)
		return dic

	# input: y_data['doc_id'] = label
	#		 datapath: train_path
	# output: line format = 'label' 'data_str'
	def prepare_svm_predict_data(self, x_data, id_path, svm_path):
		id_f = open(id_path, 'w')
		svm_f = open(svm_path, 'w')
		for doc_id, v in x_data.items():
			id_f.write('{}\n'.format(doc_id))
			svm_f.write('-1')
			for fid in sorted(v.keys()):
				svm_f.write(' {}:{}'.format(fid, v[fid]))
			svm_f.write('\n')
		id_f.close()
		svm_f.close()

	# input: y_data['doc_id'] = label
	#		 datapath: train_path
	# output: line format = 'label' 'data_str'
	def prepare_svm_data(self, y_data, output_path):
		in_f = open(self.train_data_path)
		ou_f = open(output_path, 'w')
		for line in in_f:
			doc_id, data_str = line.strip().split('\t')
			if doc_id in y_data:
				ou_f.write('{} {}\n'.format(y_data[doc_id], data_str))
		in_f.close()
		ou_f.close()

	# compute metrics
	def compute_metrics(self, y_pred):
		tp = 0
		fp = 0
		y_arr = []
		s_arr = []
		for doc_id, p in y_pred.items():
			if p > 0.5:
				pred = 1
			else:
				pred = -1
			label = self.lookup_label(doc_id)

			if label > 0:
				if pred > 0:
					tp += 1
				y_arr.append(1)
			else: # label < 0
				if pred > 0:
					fp += 1
				y_arr.append(0)
			s_arr.append(p)
		fn = self.num_pos - tp
		tn = self.num_neg - fp

		accuracy = float(tp + tn) / float(tp + tn + fp + fn)
		precision = float(tp) / float(tp + fp)
		recall = float(tp) / float(tp + fn)
		f1 = 2*precision*recall / (precision + recall)

		fpr, tpr, _ = metrics.roc_curve(y_arr, s_arr)
		auc = metrics.auc(fpr, tpr)
		ap = metrics.average_precision_score(y_arr, s_arr)
		rprec = compute_rprec(y_arr, s_arr)

		d = {'accuracy': accuracy, \
			 'precision': precision, \
			 'recall': recall, \
			 'f1': f1, \
			 'auc': auc, \
			 'ap': ap, \
			 'rprec': rprec }

		return d


if __name__ == '__main__':
	if len(sys.argv) < 4:
		exit ('Params: \n"make_eval" num_pos num_neg eval_svm_path eval_id_path\n"make_seed" num_pos num_neg seed_path')

	option = sys.argv[1]

	if option == 'make_eval':
		num_pos = int(sys.argv[2])
		num_neg = int(sys.argv[3])
		eval_dat_path = sys.argv[4]

		util = DataSetUtility()
		# make eval pool
		eval_doc_id = []
		eval_doc_id.extend( util.random_sample_docs_by_label(1,  num_pos) )
		eval_doc_id.extend( util.random_sample_docs_by_label(-1, num_neg) )
		dat = util.scan_raw_by_doc_id(eval_doc_id)
		vec = util.parse_data_to_sparse_vec(dat)
		util.write_vec_label_to_file(vec, eval_dat_path)

	if option == 'make_seed':
		num_pos = int(sys.argv[2])
		num_neg = int(sys.argv[3])
		shared_seed_path = sys.argv[4]

		util = DataSetUtility()
		# seed_doc_id = []
		# seed_doc_id.extend( util.random_sample_docs_by_label_no_eval(1,  num_pos) )
		# seed_doc_id.extend( util.random_sample_docs_by_label_no_eval(-1, num_neg) )
		seed_doc_id = ['328667', '364493', '615303', '234900']
		dat = util.scan_raw_by_doc_id(seed_doc_id)
		util.write_dat_to_file(dat, shared_seed_path)






	
		
