.. -*- mode: rst -*-

|BuildTest|_ |PyPi|_ |License|_ |Downloads|_ |PythonVersion|_

.. |BuildTest| image:: https://travis-ci.com/tank-overlord/TicTacToe4fun.svg?branch=main
.. _BuildTest: https://app.travis-ci.com/github/tank-overlord/TicTacToe4fun

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.8%20%7C%203.9-blue

.. |PyPi| image:: https://img.shields.io/pypi/v/TicTacToe4fun
.. _PyPi: https://pypi.python.org/pypi/TicTacToe4fun

.. |Downloads| image:: https://pepy.tech/badge/TicTacToe4fun
.. _Downloads: https://pepy.tech/project/TicTacToe4fun

.. |License| image:: https://img.shields.io/pypi/l/TicTacToe4fun
.. _License: https://pypi.python.org/pypi/TicTacToe4fun


================================
A Fun Experiment of Tic Tac Toe!
================================

Install
-------

.. code-block::

   pip install TicTacToe4fun


Run
---

>>> from TicTacToe4fun import game # 3x3 example below
>>> g = game()
>>> g.trials(n_trials = 1, verbosity = 0, use_hashmap = False, use_alpha_beta_pruning = False) # very slow, without using any computational technique
board_dims = (3, 3), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 3.954 sec
>>> g.trials(n_trials = 1, verbosity = 0, use_hashmap = False, use_alpha_beta_pruning = True) # α-β pruning speeds up
board_dims = (3, 3), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 0.264 sec
>>> g.trials(n_trials = 1, verbosity = 0, use_hashmap = True, use_alpha_beta_pruning = True) # initial hashmap building takes ~2MB and ~0.1s on my computer
board_dims = (3, 3), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 0.102 sec
>>> g.trials(n_trials = 10000, verbosity = 0, use_hashmap = True, use_alpha_beta_pruning = True) # afterwards, a lot faster; 10k trials completed in less than 1 sec
board_dims = (3, 3), X won #: 0, O won #: 0, Draw #: 10,000, Elapsed time: 0.860 sec
>>> g.trials(n_trials = 1, verbosity = 1, use_hashmap = True, use_alpha_beta_pruning = True) # print the board for details. 1 trial completed in less than 0.001 sec
... (board details skiped) ...
board_dims = (3, 3), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 0.000 sec

>>> from TicTacToe4fun import game # 4x4 example below
>>> g = game()
>>> g.trials(n_trials = 1, verbosity = 0, board_dims = (4, 4)) # initial hashmap building takes ~5GB and ~9min on my computer
board_dims = (4, 4), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 533.661 sec
>>> g.trials(n_trials = 10000, verbosity = 0, board_dims = (4, 4)) # afterwards, a lot faster; 10k trials completed in ~11 sec
board_dims = (4, 4), X won #: 0, O won #: 0, Draw #: 10,000, Elapsed time: 11.273 sec
>>> g.trials(n_trials = 1, verbosity = 1, board_dims = (4, 4)) # print the board for details. 1 trial completed in 0.003 sec
... (board details skiped) ...
board_dims = (4, 4), X won #: 0, O won #: 0, Draw #: 1, Elapsed time: 0.003 sec


Sample Screenshot
-----------------
|image1|


.. |image1| image:: https://github.com/tank-overlord/TicTacToe4fun/raw/main/TicTacToe4fun/examples/game1.png



Have fun!!!
-----------

