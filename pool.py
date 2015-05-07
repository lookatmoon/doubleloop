

class Pool (object):
	def __init__(self, dataset_util):
		self.dataset_util = dataset_util
		
		self.x_data = {}
		self.x_pred = {} # density estimation
		self.x_hist = {} # x_hist[doc_id] = x_query_id

		self.y_data = {}
		self.y_pred = {} # label guess
		self.y_hist = {} # y_hist[doc_id] = y_query_id

	def add_x(self, x_query_answered):
		for qid in x_query_answered:
			for doc_id, x in x_query_answered[qid].items():
				if doc_id not in self.x_data:
					self.x_data[doc_id] = x
					self.x_hist[doc_id] = qid

	def add_y(self, y_query_answered):
		for qid in y_query_answered:
			for doc_id, y in y_query_answered[qid].items():
				if doc_id not in self.y_data:
					self.y_data[doc_id] = y
					self.y_hist[doc_id] = qid

	# return: # of new +1 y_data in retrieved doc
	def num_new_positive(self):
		return 10

