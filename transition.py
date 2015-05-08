import sys
import random
import os.path
import json
from scipy.stats import spearmanr

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
			if self.y_stable():
				return self.record_state(STATE_X_QUERY)
			else:
				return self.record_state(STATE_Y_QUERY)

		elif current_step == STATE_X_UPDATE:

			if not self.y_learner.has_new_predicted_pos_data():
			# if not self.y_learner.has_new_labeled_pos_data():
				return self.record_state(STATE_DONE)
			else:
				return self.record_state(STATE_Y_UPDATE)

			# return self.record_state(STATE_Y_UPDATE)

	def get_model_dir(self):
		model_dir = os.path.join(self.pool.dataset_util.work_dir, 'model_{}_{}'.format(self.x_querier.xqid, self.y_querier.yqid))
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		return model_dir

	def save_model(self):
		model_dir = self.get_model_dir()

		# prepare model content
		m = {}
		m['query_arr'] = []
		for h in self.x_querier.hist:
			q_str = self.pool.dataset_util.parse_sparse_vec_to_query_str(h[2])
			p = {'q': q_str}
			p['r'] = 1000
			m['query_arr'].append(p)
		
		if self.y_learner.current_model_path != None:
			m['classifier'] = os.path.abspath(self.y_learner.current_model_path)
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

	def y_stable(self):
		n_y_last_pred = len(self.y_learner.y_last_pred)
		n_y_pred = len(self.pool.y_pred)
		if n_y_last_pred != n_y_pred:
			return False
		y_last_pred_score = []
		y_pred_score = []
		for doc_id in self.pool.y_pred:
			y_last_pred_score.append(self.y_learner.y_last_pred[doc_id])
			y_pred_score.append(self.pool.y_pred[doc_id])
		r, p = spearmanr(y_last_pred_score, y_pred_score)
		# sys.stderr.write('YLearner, is_stable(): y_last_pred_score = {}\n'.format(self.y_last_pred))
		# sys.stderr.write('YLearner, is_stable(): y_pred_score = {}\n'.format(self.pool.y_pred))
		sys.stderr.write('Transition, y_stable(): spearmanr = {}\n'.format(r))
		if r > 0.85:
			return True
		else:
			return False

	def x_stable(self):
		if self.pool.x_new == 0:
			return True
		else:
			return False



