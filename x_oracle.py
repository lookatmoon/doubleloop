
class XOracle(object):

	def __init__(self, dataset_util):
		self.dataset_util = dataset_util

	def get_seeds(self):
		seed_xqid = 0
		dat = self.dataset_util.read_dat_from_file(self.dataset_util.x_seed_path)
		dat = self.dataset_util.restrict_id(dat) # excluding data in eval
		vec = self.dataset_util.parse_data_to_sparse_vec(dat)
		return {seed_xqid : vec}

	# input: x_queries[xqid] = sparse_query_vec
	# output: answer[xqid][doc_id] = sparse_vec
	# that means multiple searches is possible
	def answer(self, x_queries):
		for qid in x_queries:
			x_query_vec = x_queries[qid]
			query_str = self.dataset_util.parse_sparse_vec_to_query_str(x_query_vec)
			params = {'max':500, 'query': query_str}
			dat = self.dataset_util.search_index(params)
			dat = self.dataset_util.restrict_id(dat) # excluding data in eval
			vec = self.dataset_util.parse_data_to_sparse_vec(dat)
			x_queries[qid] = vec
		return x_queries

if __name__ == "__main__":
	from dataset_utility import *

	util = DataSetUtility()
	x_oracle = XOracle(util)
	print x_oracle.get_seeds()
	# print util.term2id
	# print util.id2term
	# print x_oracle.answer({'43647':1, '26599':1})