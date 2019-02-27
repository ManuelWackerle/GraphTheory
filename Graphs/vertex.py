from typing import List, Union, Set
from Graphs.graph import GraphError

class Vertex(object):
    """
    `Vertex` objects have a property `graph` pointing to the graph they are part of,
    and an attribute `label` which can be anything: it is not used for any methods,
    except for `__str__`.
    """

    def __init__(self, graph: "MyGraph", label=None):
        """
        Creates a vertex, part of `graph`, with optional label `label`.
        (Labels of different vertices may be chosen the same; this does
        not influence correctness of the methods, but will make the string
        representation of the graph ambiguous.)
        :param graph: The graph that this `Vertex` is a part of
        :param label: Optional parameter to specify a label for the
        """
        if label is None:
            label = graph._next_label()

        self._graph = graph
        # self.colortext = '#00000'
        self.colornum = 0
        self.colornew = 0
        self.label = label
        self._incidence = {}


    def __repr__(self):
        """
        A programmer-friendly representation of the vertex.
        :return: The string to approximate the constructor arguments of the `Vertex'
        """
        return 'Vertex(label={}, #incident={})'.format(self.label, len(self._incidence))

    def __str__(self) -> str:
        """
        A user-friendly representation of the vertex, that is, its label.
        :return: The string representation of the label.
        """
        return str(self.label)

    def is_adjacent(self, other: "Vertex") -> bool:
        """
        Returns True iff `self` is adjacent to `other` vertex.
        :param other: The other vertex
        """
        return other in self._incidence

    def _add_incidence(self, edge: "Edge"):
        """
        For internal use only; adds an edge to the incidence map
        :param edge: The edge that is used to add the incidence
        """
        other = edge.other_end(self)

        if other not in self._incidence:
            self._incidence[other] = set()

        self._incidence[other].add(edge)

    def add_incidence(self, edge: "Edge"):
        self._add_incidence(edge)

    def _remove_incidence(self, edge: "Edge"):
        """
        ADDED THIS METHOD. NOT SURE IF IT IS NECESSARY!?!?
        """
        other = edge.other_end(self)
        if other in self._incidence:
            self._incidence.pop(other)

    def remove_incidence(self, edge: "Edge"):
        self._remove_incidence(edge)

    @property
    def graph(self) -> "MyGraph":
        """
        The graph of this vertex
        :return: The graph of this vertex
        """
        return self._graph

    @property
    def incidence(self) -> List["Edge"]:
        """
        Returns the list of edges incident with the vertex.
        :return: The list of edges incident with the vertex
        """
        result = set()

        for edge_set in self._incidence.values():
            result |= edge_set

        return list(result)

    @property
    def neighbours(self) -> List["Vertex"]:
        """
        Returns the list of neighbors of the vertex.
        """
        return list(self._incidence.keys())

    @property
    def degree(self) -> int:
        """
        Returns the degree of the vertex
        """
        return sum(map(len, self._incidence.values()))
