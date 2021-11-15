# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

from TicTacToe4fun import game
g = game()
#
g.trials(n_trials = 1,     verbosity = 1, board_dims = (3, 3), use_hashmap = False, use_alpha_beta_pruning = False)
g.trials(n_trials = 1,     verbosity = 0, board_dims = (3, 3), use_hashmap = False, use_alpha_beta_pruning = True )
g.trials(n_trials = 1,     verbosity = 0, board_dims = (3, 3), use_hashmap = True,  use_alpha_beta_pruning = False)
g.trials(n_trials = 1,     verbosity = 0, board_dims = (3, 3), use_hashmap = True,  use_alpha_beta_pruning = True )
g.trials(n_trials = 10000, verbosity = 0, board_dims = (3, 3), use_hashmap = True,  use_alpha_beta_pruning = True )
#
#g.trials(n_trials = 1,      verbosity = 1, board_dims = (4, 4))
#g.trials(n_trials = 10000,  verbosity = 0, board_dims = (4, 4))
