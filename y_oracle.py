
class YOracle(object):

	def __init__(self, dataset_util):
		self.dataset_util = dataset_util

	# input: y_queries[qid][doc_id] = None
	# output: answer[qid][doc_id] = label
	def answer(self, y_queries):
		for qid in y_queries:
			for doc_id in y_queries[qid]:
				label = self.dataset_util.lookup_label(doc_id)
				y_queries[qid][doc_id] = label
		print y_queries
		return y_queries
