from datetime import *
import sys
import operator
import random

class YQuerier(object):
	def __init__(self, pool):
		self.pool = pool
		self.hist = [] # (timestamp, y_query_id, y_query)
		self.yqid = 0

	def ask_seeds(self):
		# ask all
		seed_yqid = 0
		y_queries = {seed_yqid : {}}
		for doc_id in self.pool.x_data:
			y_queries[seed_yqid][doc_id] = None
			self.hist.append( (datetime.now(), seed_yqid, doc_id) )
		return y_queries

	def ask_margin(self, k_select):
		y_queries = {}
		self.yqid += 1
		y_queries[self.yqid] = {}

		score_dict = {}
		for doc_id in self.pool.x_data:
			if (doc_id not in self.pool.y_data) and (doc_id in self.pool.y_pred):
				# sys.stderr.write('YQuerier, ask_margin(): doc_id in y_pred\n')
				score_dict[doc_id] = abs(self.pool.y_pred[doc_id] - 0.5)
		score_sorted = sorted(score_dict.items(), key = operator.itemgetter(1))
		
		for i in range(min( k_select, len(score_sorted) )):
			doc_id = score_sorted[i][0]
			y_queries[self.yqid][doc_id] = None
			self.hist.append( (datetime.now(), self.yqid, doc_id) )
			sys.stderr.write('YQuerier, ask_margin(): y_query = {}, margin = {}\n'.format(doc_id, score_sorted[i][1]))

		return y_queries

	# return: y_queries[qid][doc_id] = None
	def ask(self):
		k_select = 10
		return self.ask_margin(k_select)
