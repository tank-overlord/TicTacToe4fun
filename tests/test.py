# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

from TicTacToe4fun import play
game = play()
game.trials(n_trials=10000)
game.trials(verbosity=1, n_trials=1)

