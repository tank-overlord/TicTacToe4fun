# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT


from .__about__ import (
    __version__,
    __license__,
)

from .tic_tac_toe import play

# this is for "from <package_name> import *"
__all__ = ["play", ]



