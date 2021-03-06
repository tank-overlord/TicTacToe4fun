# -*- coding: utf-8 -*-

#  Author: Tank Overlord <TankOverLord88@gmail.com>
#
#  License: MIT

import setuptools

import TicTacToe4fun

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="TicTacToe4fun",
    version=TicTacToe4fun.__version__,
    author="Tank Overlord",
    author_email="TankOverLord88@gmail.com",
    description="A Fun Experiment of Tic-Tac-Toe!",
    license=TicTacToe4fun.__license__,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/tank-overlord/TicTacToe4fun",
    packages=setuptools.find_packages(),
    # https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.8',
    include_package_data=True,
)
