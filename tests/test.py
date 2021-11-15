# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

from TicTacToe4fun import game
g = game()
#
g.trials(n_trials = 1,  verbosity = 0, use_hashmap = False, use_alpha_beta_pruning = False) # very slow, without using any computational technique
g.trials(n_trials = 1,  verbosity = 0, use_hashmap = False, use_alpha_beta_pruning = True) # α-β pruning speeds up
g.trials(n_trials = 1,  verbosity = 0, use_hashmap = True, use_alpha_beta_pruning = True) # initial hashmap building takes ~2MB and 0.1s on my computer
g.trials(n_trials = 10000,  verbosity = 0, use_hashmap = True, use_alpha_beta_pruning = True) # afterwards, a lot faster; 10k trials completed in less than 1 sec
g.trials(n_trials = 1,  verbosity = 1, use_hashmap = True, use_alpha_beta_pruning = True) # print the board for details. 1 trial completed in less than 0.001 sec
#
#g.trials(n_trials = 1,      verbosity = 1, board_dims = (4, 4))
#g.trials(n_trials = 10000,  verbosity = 0, board_dims = (4, 4))
