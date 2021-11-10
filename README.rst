.. -*- mode: rst -*-

|BuildTest|_ |PyPi|_ |License|_ |Downloads|_ |PythonVersion|_

.. |BuildTest| image:: https://travis-ci.com/tank-overlord/TicTacToe4fun.svg?branch=main
.. _BuildTest: https://travis-ci.com/tank-overlord/TicTacToe4fun

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.8%20%7C%203.9-blue

.. |PyPi| image:: https://img.shields.io/pypi/v/TicTacToe4fun
.. _PyPi: https://pypi.python.org/pypi/TicTacToe4fun

.. |Downloads| image:: https://pepy.tech/badge/TicTacToe4fun
.. _Downloads: https://pepy.tech/project/TicTacToe4fun

.. |License| image:: https://img.shields.io/pypi/l/TicTacToe4fun
.. _License: https://pypi.python.org/pypi/TicTacToe4fun


===============================
A Fun Experiment of Tic Tac Toe
===============================

Install
-------

.. code-block::

   pip install TicTacToe4fun


Execute
-------

.. code-block::

>>> from TicTacToe4fun import play
>>> game = play()
>>> game.trials()
>>> game.trials(verbosity=1, n_trials=1)
--------
Game#1

|   |
|   |
|   |

| O |
|   |
|   |

| O |
| X |
|   |

| O |
| X |
|O  |

|XO |
| X |
|O  |

|XO |
| X |
|O O|

|XO |
| X |
|OXO|

|XOO|
| X |
|OXO|

|XOO|
| XX|
|OXO|

|XOO|
|OXX|
|OXO|

X won #: 0, O won #: 0, Draw #: 1


Have fun!!!
-----------

