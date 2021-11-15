# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

from TicTacToe4fun import game
g = game()
g.trials(n_trials=10000)
g.trials(verbosity=1, n_trials=1)

