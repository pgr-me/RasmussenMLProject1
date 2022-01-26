#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, datamaker.py

The DataMaker class creates a random sequence of binary integers.

"""

# Standard library imports
from typing import List


class DataMaker:
    """
    This class generates pairs of pseudo-random numbers using the linear congruential generator algorithm.
    Sources:
    Parameters based on https://en.wikipedia.org/wiki/Linear_congruential_generator
    make_pseudo_random method adapted from https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
    """

    def __init__(self, n, seed=777, a=1103515245, c=12345, m=2 ** 15):
        """
        Constructor.
        :param n: Number of pairs to generate
        :param seed: Seed of pseudo-random sequence
        :param a: Multiplier
        :param c: Increment
        :param m: Modulus
        """
        self.n = n
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.points = []
        self.data = []

    def make_data(self) -> List[int]:
        """
        Make a sequence of pseudo-random x-y pairs.
        :return: List of x-y pairs
        """
        seed = self.seed
        for i in range(self.n):
            seed = self.make_pseudo_random(self.a, self.c, self.m, seed)
            is_one = (self.a * seed + self.c) % self.m / self.m >= 0.5
            self.data.append(1 if is_one else 0)

        return self.data

    @staticmethod
    def make_pseudo_random(a: int, c: int, m: int, seed: int) -> int:
        """
        Make a pseudo-random binary integer.
        :param a: Multiplier
        :param c: Increment
        :param m: Modulus
        :param seed: Seed to initiate pseudo-random sequence
        :return: Pseudo-random binary integer
        """
        return (a * seed + c) % m

