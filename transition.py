import sys
import random
import os.path
import json
import liblinearutil as linsvm

STATE_X_QUERY  = 'STATE_X_QUERY'
STATE_X_UPDATE = 'STATE_X_UPDATE'
STATE_Y_QUERY  = 'STATE_Y_QUERY'
STATE_Y_UPDATE = 'STATE_Y_UPDATE'
STATE_DONE     = 'STATE_DONE'

STRATEGY_RANDOM = 'STRATEGY_RANDOM'
STRATEGY_DOUBLE_LOOP = 'STRATEGY_DOUBLE_LOOP'

class Transition(object):
	def __init__(self, pool, x_querier, x_learner, y_querier, y_learner):
		self.pool = pool
		self.x_querier = x_querier
		self.x_learner = x_learner
		self.y_querier = y_querier
		self.y_learner = y_learner
		self.step_hist = []

		# self.strategy = 'STRATEGY_RANDOM'
		self.strategy = 'STRATEGY_DOUBLE_LOOP'

	# check if the y_learner generated stable statistics.
	# def y_learner_is_stable(self):
		# pass

	# check if the x_querier retrieved 
	# def x_learner_is_standing(self):
		# pass

	def record_state(self, state):
		self.step_hist.append(state)
		return state


	# check current status, output: next step
	def go(self):
		if self.strategy == STRATEGY_RANDOM:
			return self.go_random()

		elif self.strategy == STRATEGY_DOUBLE_LOOP:
			return self.go_double_loop()

	def go_random(self):
		next_step = random.choice([STATE_X_QUERY, STATE_X_UPDATE, STATE_Y_QUERY, STATE_Y_UPDATE])
		self.step_hist.append(next_step)
		return next_step

	def go_double_loop(self):
		if len(self.step_hist) == 0: # initial step
			return self.record_state(STATE_X_QUERY)

		current_step = self.step_hist[-1]

		if current_step == STATE_X_QUERY:
			return self.record_state(STATE_X_UPDATE)

		elif current_step == STATE_Y_QUERY:
			return self.record_state(STATE_Y_UPDATE)

		elif current_step == STATE_Y_UPDATE:
			if self.y_learner.is_stable():
				return self.record_state(STATE_X_QUERY)
			else:
				return self.record_state(STATE_Y_QUERY)

		elif current_step == STATE_X_UPDATE:
			if not self.y_learner.has_new_predicted_pos_data():
			# if not self.y_learner.has_new_labeled_pos_data():
				return self.record_state(STATE_DONE)
			else:
				return self.record_state(STATE_Y_UPDATE)

	def save_model(self):
		model_dir = os.path.join(self.pool.dataset_util.work_dir, 'model_{}_{}'.format(self.x_querier.xqid, self.y_querier.yqid))
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		# prepare model content
		m = {}
		m['query_arr'] = []
		for h in self.x_querier.hist:
			q_str = self.pool.dataset_util.parse_sparse_vec_to_query_str(h[2])
			p = {'q': q_str}
			p['r'] = 500
			m['query_arr'].append(p)
		
		if self.y_learner.model != None:
			model_path = os.path.join(model_dir, 'model')
			linsvm.save_model(model_path, self.y_learner.model)
			m['classifier'] = os.path.abspath(model_path)
		else:
			m['classifier'] = 'NULL'

		# collect current status: #y_data, #(+1)y_data, #x_data
		m['num_y'] = len(self.pool.y_data)
		m['num_pos_y'] = sum([1 for doc_id, l in self.pool.y_data.items() if l > 0])
		m['num_x'] = len(self.pool.x_data)

		json_path = os.path.join(model_dir, 'model.json')
		f = open(json_path, 'w')
		json.dump(m, f, sort_keys=True, indent=2)
		f.close()





