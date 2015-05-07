import sys

class XLearner(object):

	def __init__(self, pool):
		self.pool = pool

		# memory of the past
		self.last_num_x_data = 0

	def update_belief(self):
		pass