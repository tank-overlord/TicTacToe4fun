# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT
 
import sys

from .tic_tac_toe import play

if __name__ == "__main__":
    sys.exit(play().trials(verbosity=1))
