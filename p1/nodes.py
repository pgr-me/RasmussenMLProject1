#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, nodes/nodes.py

This module provides the base Node class and the RootNode class, which inherits from the former.

"""

# Standard library imports
import collections as c
import typing as t


class Node:
    """Base node class used for signal processing."""

    def __init__(self, i: int, j: int, s: t.List[int], x, y):
        self.i = i
        self.j = j
        self.s = s
        self.x = x
        self.y = y
        self.x_match = None
        self.y_match = None
        self.length = None
        self.key = None
        self.left = None
        self.right = None
        self.up = None
        self.rank = 0

    def __repr__(self):
        return f"Node i={self.i}, j={self.j}"

    def comp(self):
        """
        Compare x and y to s.
        """
        self.x_match = (self.x == self.s) and self.x not in [None, "$$"]
        self.y_match = self.y == self.s and self.y not in [None, "$$"]

    def set_key(self):
        """
        Set node key as (i, j) pair.
        """
        self.key = (self.i, self.j)

    def is_match(self) -> bool:
        """
        Check if x or y matches corresponding s char.
        :return: True if either x or y matches
        """
        return self.x_match or self.y_match


class RootNode(Node):
    """Root node of Graph."""
    def __init__(self):
        super().__init__(i=0, j=0, s=None, x=None, y=None)
        self.adj = {}
        self.ranks = c.defaultdict(list)

    def __repr__(self):
        return f"Root Node"