from random import *
from datetime import *
import operator

def sum_coeff_vec(coeff_dict, vec_dict):
	s = {}
	for doc_id in coeff_dict:
		c = coeff_dict[doc_id]
		if doc_id in vec_dict:
			for dim_id, val in vec_dict[doc_id].items():
				if dim_id not in s:
					s[dim_id]  = c * val
				else:
					s[dim_id] += c * val
	return s

class XQuerier(object):

	def __init__(self, pool):
		self.pool = pool
		self.hist = [] # (timestamp, x_query_id, x_query)

		self.xqid = 0

	def ask_rocchio(self, cutoff, alpha, beta, gamma):
		x_queries = {}

		# prepare coefficients for each data item
		num_pos = sum([1.0 for doc_id, label in self.pool.y_data.items() if label > 0])
		num_neg = sum([1.0 for doc_id, label in self.pool.y_data.items() if label < 0])
		beta /= num_pos
		gamma /= num_neg
		coeff = {}
		for doc_id, label in self.pool.y_data.items():
			if label == 1:
				coeff[doc_id] = beta
			elif label == -1:
				coeff[doc_id] = -gamma
		vec = sum_coeff_vec(coeff, self.pool.x_data)

		sorted_vec = sorted(vec.iteritems(), key=operator.itemgetter(1), reverse=True)
		x_query = dict(sorted_vec[:cutoff])

		self.xqid += 1
		self.hist.append( (datetime.now(), self.xqid, x_query) )

		x_queries[self.xqid] = x_query
			
		return x_queries

	# output: x_queries[qid] = sparse_vec
	def ask(self):
		return self.ask_rocchio(4, 1.0, 1.0, 0.2)
		# return self.ask_diverse_rocchio()


