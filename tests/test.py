# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

from TicTacToe4fun import game
g = game()
#
g.trials(n_trials = 1,      verbosity = 1, board_dims = (3, 3))
g.trials(n_trials = 10000,  verbosity = 0, board_dims = (3, 3))
#
g.trials(n_trials = 1,      verbosity = 1, board_dims = (4, 4))
g.trials(n_trials = 10000,  verbosity = 0, board_dims = (4, 4))
