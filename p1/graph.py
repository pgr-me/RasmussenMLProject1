#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, graph.py

Graph provides the means to compute the longest interweaving of signals x and y in s.

"""

# Standard library imports
import collections as c
import typing as t

# Local imports
from pa2.nodes import Node, RootNode
from pa2.signal_processing import get_s_ij, get_x_ij, get_y_ij, prune_path


class Graph:
    """Instantiates a graph object to find the longest path in the candidate interweaving."""

    def __init__(self, s: t.List[int], x: t.List[int], y: t.List[int]):
        self.s = s
        self.x = x
        self.y = y
        self.root = RootNode()
        self.T = []
        self.max_rank_node = None
        self.path = {}
        self.n_comps = 0
        self.n_ops = 0

    def add_adj(self, node: Node):
        """
        Add adjacent node to root node.
        :param node: Node to add to root
        """
        self.root.adj[(node.i, node.j)] = node
        self.n_ops += 1

    def build_graph(self):
        """
        Build graph of linked nodes.
        """
        # Iterate over reach row, excluding last one since we don't need it
        for i in range(len(self.s) - 1):

            # Iterate over each column, excluding last one since we don't need it
            for j in range(len(self.s) - 1):

                # Only process elements that are in upper triangular portion of T
                if i + j < len(self.s):

                    # Acquire values at each string location
                    node = self.T[i][j]

                    # If there is an x or y match, process the node
                    if node.x_match or node.y_match:
                        self.add_adj(node)
                        node.rank = 1
                        self.link(node)
                        if node.left and node.left.key in self.root.adj:
                            left_node_rank = node.left.rank
                        else:
                            left_node_rank = 0
                        if node.up and node.up.key in self.root.adj:
                            up_node_rank = node.up.rank
                        else:
                            up_node_rank = 0
                        node.rank += max(left_node_rank, up_node_rank)

    def build_table(self) -> t.List[list]:
        """
        Build table of instantiated - but unlinked - nodes.
        :return: List of lists
        """
        for i in range(len(self.s)):
            temp = []
            for j in range(len(self.s)):
                s_ij = get_s_ij(self.s, i, j)
                x_ij = get_x_ij(self.x, self.s, i, j)
                y_ij = get_y_ij(self.y, self.s, i, j)
                node = Node(i, j, s_ij, x_ij, y_ij)
                node.set_key()
                node.comp()
                temp.append(node)
            self.T.append(temp)
        self.get_length()
        return self.T

    def display(self, attr: str = "rank") -> t.List[list]:
        """
        Display table of selected attribute values.
        :param attr: Selected attribute
        :return: Table of selected attribute values
        """
        table = []
        for i in range(self.length):
            temp = []
            for j in range(self.length):
                temp.append(getattr(self.T[i][j], attr))
            table.append(temp)
        return table

    def find_max_rank_node(self) -> int:
        """
        Find node with highest rank.
        :return: Max rank node
        """
        max_rank = -1
        for key, node in self.root.adj.items():
            if node.rank > max_rank:
                self.max_rank_node = node
                max_rank = node.rank
        return self.max_rank_node

    def find_path(self, node: Node) -> dict:
        """
        Find path from node to root.
        :param node: Node to find path from
        :return: Dictionary of linked nodes that comprise path
        """
        rank = node.rank

        # Return the empty path if node rank is zero
        if node.rank == 0:
            self.n_ops += 1
            return self.path

        # If max rank is only 1, we simply set the path as that one node
        self.n_ops += 1
        self.path = {rank: {"node": node,
                            "signal": "x" if node.x_match else "y",
                            "s_ix": node.i + node.j,
                            "s_char": node.s}}
        # If max node rank is greater than 1, we iterate over qualifying nodes by rank
        if node.rank > 1:

            while True:

                # If there is a left- or up-node, then add that to the path
                if node.left or node.up:
                    if node.left:
                        node = node.left
                        signal = "x"
                    else:
                        node = node.up
                        signal = "y"
                    self.path[node.rank] = {"node": node,
                                            "signal": signal,
                                            "s_ix": node.i + node.j,
                                            "s_char": node.s}
                    self.n_ops += 2  # One for if / else, the other for adding node to dict
                    self.n_comps += 1  # One for the if / else

                # Otherwise, terminate the while loop when node rank == 1
                else:
                    self.n_ops += 1
                    break
        self.n_ops += 1
        return self.path

    def find_longest_path(self) -> dict:
        """
        Find longest un-pruned path in graph.
        :return: Longest un-pruned path
        """
        ranks = c.defaultdict(list)
        self.longest_path = {}
        path_nodes = set()
        self.n_ops += 3  # Three for the three lines above

        # Build the ranks dictionary which organizes nodes by ranks
        for ij, node in self.root.adj.items():
            ranks[node.rank].append(node)
            self.n_ops += 2  # Two for the above 2 lines

        # Iterate over each rank from highest to lowest
        for rank in reversed(sorted(ranks.keys())):
            self.n_ops += 1
            # Return empty longest path in case when rank == 0
            if rank == 0:
                self.n_ops += 2
                return self.longest_path

            # If the longest path is already longer than the current rank, return path
            if len(self.longest_path) > rank:
                self.n_ops += 2
                return self.longest_path

            # Iterate over each node in the ranks dictionary
            for ix, node in enumerate(ranks[rank]):
                self.n_ops += 1  # One operation each time above line executed

                # We only need to process highest-ranked node in path
                # Only process those highest-ranked nodes that haven't already been included in path_nodes set
                self.n_ops += 1  # One for if statement below
                if ix == 0 and node not in path_nodes:

                    # Add path items to path_nodes set so we only traverse unique trees
                    # n_ops and n_comps accounted for in find_path method
                    path = self.find_path(self.T[node.i][node.j])
                    for rank, di in path.items():
                        path_nodes.add(di["node"])
                        self.n_ops += 1

                    # Prune path items that don't satisfy requirement that x and y repetitions be complete
                    path, prune_ops = prune_path(path, self.x, self.y)
                    self.n_ops += prune_ops
                    if len(path) > len(self.longest_path):
                        self.longest_path = path
                        self.n_ops += 2  # Two ops: one for if line and the other for self.longest_path = path line

        self.n_ops += 1  # One for return statement
        return self.longest_path

    def get_length(self) -> int:
        """
        Compute length of table T, which is square.
        :return: Integer length of table
        """
        self.length = len(self.T)
        self.n_ops += 2  # Two operations, one for the above line and one for the below line
        return self.length

    def link(self, node: Node):
        """
        Link node to adjacent nodes.
        :param node: Node to link
        """
        i, j = node.i, node.j
        right_node = self.T[i][j + 1]
        down_node = self.T[i + 1][j]
        make_x_link = node.x_match and right_node.is_match()
        make_y_link = node.y_match and down_node.is_match()
        if make_x_link:
            node.right = right_node
            right_node.left = node
        if make_y_link:
            node.down = down_node
            down_node.up = node
        self.n_comps += 2
        self.n_ops += 11
