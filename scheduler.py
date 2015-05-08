import sys

if len(sys.argv) != 2:
	exit ('Params: id_set_path')
id_set_path = sys.argv[1]

from dataset_utility import DataSetUtility
from pool import Pool

from x_oracle import XOracle
from x_querier import XQuerier
from x_learner import XLearner

from y_oracle import YOracle
from y_querier import YQuerier
from y_learner import YLearner

from transition import *

util = DataSetUtility(id_set_path)
pool = Pool(util)

# X's
x_oracle  = XOracle(util)
x_querier = XQuerier(pool)
x_learner = XLearner(pool)

# Y's
y_oracle  = YOracle(util)
y_querier = YQuerier(pool)
y_learner = YLearner(pool)

pool.transition = Transition(pool, x_querier, x_learner, y_querier, y_learner)

# initialize
x_seeds = x_oracle.get_seeds()
print 'INITIAL x_seeds', len(x_seeds)
pool.add_x(x_seeds)
print 'INITIAL x_data', len(pool.x_data)
print 'INITIAL x_hist', len(pool.x_hist)

y_seeds = y_querier.ask_seeds()
y_seeds_answered = y_oracle.answer(y_seeds)
pool.add_y(y_seeds_answered)
print 'INITIAL y_data', len(pool.y_data)
print 'INITIAL y_hist', len(pool.y_hist)
print 'INITIAL y_pred', len(pool.y_pred)

next_state = pool.transition.go()

while True:
	print 'current_state =', next_state
	if   next_state == STATE_Y_UPDATE:
		y_learner.update_belief()

	elif next_state == STATE_X_UPDATE:
		x_learner.update_belief()

	elif next_state == STATE_Y_QUERY:
		y_query = y_querier.ask()
		y_query_answered = y_oracle.answer(y_query)
		pool.add_y(y_query_answered)

	elif next_state == STATE_X_QUERY:
		x_query = x_querier.ask()
		x_query_answered = x_oracle.answer(x_query)
		pool.add_x(x_query_answered)

	elif next_state == STATE_DONE:
		break

	else:
		exit('Error: Illegal next_state: "{}".'.format(next_state))

	if len(pool.y_data) > 300: # stop criteria
		break

	pool.transition.save_model()
	next_state = pool.transition.go()

