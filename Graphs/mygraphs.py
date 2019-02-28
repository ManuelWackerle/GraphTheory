from Graphs.vertex import Vertex
from Graphs.edge import Edge
from typing import IO, Tuple, List, Union, Set
from time import time
from Graphs.graph_io import *
import random
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt

class MyGraph(object):
    def __init__(self, directed: bool, n: int=0, simple: bool=False):
        self._v = list()
        self._size = 0
        self._e = list()
        self._simple = simple
        self._directed = directed
        self._next_label_value = 0
        for i in range(n):
            self.add_vertex(Vertex(self))

    def __repr__(self):
        return 'Graph(directed={}, simple={}, #edges={n_edges}, #vertices={n_vertices})'.format(
            self._directed, self._simple, n_edges=len(self._e), n_vertices=len(self._v))

    def __str__(self) -> str:
        return 'V=[' + ", ".join(map(str, self._v)) + ']\nE=[' + ", ".join(map(str, self._e)) + ']'

    def _next_label(self) -> int:
        result = self._next_label_value
        self._next_label_value += 1
        return result

    @property
    def simple(self) -> bool:
        return self._simple

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def vertices(self) -> List["Vertex"]:
        return list(self._v)

    @property
    def edges(self) -> List["Edge"]:
        return list(self._e)

    def __iter__(self):
        return iter(self._v)

    def __len__(self) -> int:
        return len(self._v)

    def add_vertex(self, vertex: "Vertex"):
        if vertex.graph != self:
            raise GraphError("A vertex must belong to the graph it is added to")
        self._size += 1
        self._v.append(vertex)

    def add_edge(self, edge: "Edge"):
        if self._simple:
            if edge.tail == edge.head:
                raise GraphError('No loops allowed in simple graphs')

            if self.is_adjacent(edge.tail, edge.head):
                raise GraphError('No multiedges allowed in simple graphs')

        if edge.tail not in self._v:
            self.add_vertex(edge.tail)
        if edge.head not in self._v:
            self.add_vertex(edge.head)

        self._e.append(edge)
        edge.head._add_incidence(edge)
        edge.tail._add_incidence(edge)
            

    def complement(self, ignore_not_simple: bool = False):
        if ignore_not_simple or self.simple:
            C = MyGraph(self.directed, self._size)
            for i in range(0, self._size):
                for j in range(0, self._size):
                    if i != j and not self.vertices[i].is_adjacent(self.vertices[j]):
                        if self.directed or not C.vertices[i].is_adjacent(C.vertices[j]):
                            C.add_edge(Edge(C.vertices[i], C.vertices[j]))
        else:
            raise GraphError("The complement exists only for simple graphs")
        return C

    def __add__(self, other: "MyGraph") -> "MyGraph":
        g_new = MyGraph(self.directed, 0, self.simple)
        iso1, iso2 = {}, {}
        for i in range(0, len(self.vertices)):
            g_new.add_vertex(Vertex(g_new))
            iso1.update({self.vertices[i]: g_new.vertices[i]})
        for v in other.vertices:
            v_new = Vertex(g_new)
            g_new.add_vertex(v_new)
            iso2.update({v: v_new})
        for e in self.edges:
            new_e = Edge(iso1[e.tail], iso1[e.head])
            g_new.add_edge(new_e)
        for e in other.edges:
            new_e = Edge(iso2[e.tail], iso2[e.head])
            g_new.add_edge(new_e)
        return g_new

    def __iadd__(self, other: Union[Edge, Vertex]) -> "MyGraph":
        if isinstance(other, Vertex):
            self.add_vertex(other)
        if isinstance(other, Edge):
            self.add_edge(other)

        return self

    def find_edge(self, u: "Vertex", v: "Vertex") -> Set["Edge"]:
        result = u._incidence.get(v, set())
        if not self._directed:
            result |= v._incidence.get(u, set())
        return set(result)

    def is_adjacent(self, u: "Vertex", v: "Vertex") -> bool:
        return v in u.neighbours and (not self.directed or any(e.head == v for e in u.incidence))

    def del_edge(self, edge):
        if edge in self._e:
            edge.head.remove_incidence(edge)
            edge.tail.remove_incidence(edge)
            self._e.remove(edge)

    def del_vertex(self, vertex):
        if vertex in self._v:
            for edge in vertex.incidence:
                self.del_edge(edge)
            self._size -= 1
            self._v.remove(vertex)

    def deepcopy(self):
        copy = MyGraph(self.directed, 0, self.simple)
        iso = {}
        for i in range(0, len(self.vertices)):
            self.add_vertex(Vertex(self, v.label))
            iso.update({self.vertices[i]: copy.vertices[i]})
        for e in self.edges:
            new_e = Edge(iso[e.tail], iso[e.head])
            copy.add_edge(new_e)

    def mapcopy(self, map):
        newmap = {}
        for k in map.keys():
            nk = deepcopy(k)
            na = []
            for v in map.get(k):
                na.append(v)
            newmap.update({nk: na})
        return newmap


    def contains_cycle(self):

        if self.directed:
            pass
        else:
            for v in self.vertices:
                pass

    def is_isomorphic_by_colour(self, graph):
        length = len(self.vertices)
        n_graph = self + graph
        if len(graph.vertices) != length:
            return False, n_graph, None
        size = length * 2
        partition = []
        for i in range(0,size):
            partition.append([])
        deg_list = [-1] * (size - 1)
        for u in n_graph.vertices:
            ud = u.degree
            deg_list[ud] = ud
        self.relabel_col(deg_list)
        label = 1
        for v in n_graph.vertices:
            vd = v.degree
            label = deg_list[vd]
            v.colornum = label
            v.colornew = label
            partition[label].append(v)
        point = label
        n_point = point + 1
        while point != n_point:
            point = n_point
            iter = 0
            while iter < n_point:
                col_group = partition[iter]
                iter += 1
                width = len(col_group)
                if width > 1:
                    type = self.neighbour_colours(col_group[0])
                    first = True
                    for i in range(1, width):
                        v = col_group[i]
                        if self.neighbour_colours(v) != type:
                            if first:
                                nc = n_point
                                n_point += 1
                                first = False
                            v.colornew = nc
                    k = 1
                    for j in range(1, width):
                        y = col_group[k]
                        nc = y.colornew
                        if y.colornum != nc:
                            col_group.remove(y)
                            partition[nc].append(y)
                            y.colornum = nc
                        else:
                            k += 1
        for pair in partition:
            l = len(pair)
            if l == 1:
                return False, n_graph
            if l > 2:
                return "undetermined", n_graph
        return True, n_graph


    def relabel_col(self, col_list):
        n_label = 0
        for i in range(0, len(col_list)):
            if col_list[i] != -1:
                col_list[i] = n_label
                n_label += 1

    def neighbour_colours(self, vertex: Vertex):
        group = vertex.neighbours
        gcolour = []
        for g in group:
            gcolour.append(g.colornum)
        s1 = sorted(gcolour)
        return s1

    def construct_adjacency_matrix(self):
        adj_matrix = []
        i = 0
        for v1 in self.vertices:
            adj_matrix.append([])
            for v2 in self.vertices:
                if v1 != v2 and v1.is_adjacent(v2):
                    adj_matrix[i].append(1)
                else:
                    adj_matrix[i].append(0)
            i += 1
        return adj_matrix

    def degree_list(self, adj_matrix):
        deg_list = []
        size = len(adj_matrix)
        for i in range(0, size):
            deg = 0
            for j in range(0, size):
                deg += adj_matrix[i][j]
            deg_list.append(deg)
        return deg_list

    def construct_incidence_matirx(self):
        inc_matrix = []
        i = 0
        for v in self.vertices:
            inc_matrix.append([])
            for e in self.edges:
                if e.incident(v):
                    inc_matrix[i].append(1)
                else:
                    inc_matrix[i].append(0)
            i += 1
        return inc_matrix

    def uninformed_search_weighted(self, v: "Vertex", dfs: bool=False):
        frontier = [(self.vertices[0], 0)]
        closed = []
        while len(frontier) != 0:
            if dfs: current = frontier.pop()
            else: current = frontier.pop(0)
            w_depth = 0
            cur_ver = current[0]
            for vert in cur_ver.neighbours:
                min_edge = min([e.weight for e in self.find_edge(cur_ver, vert)])
                if min_edge is None:
                    min_edge = 1
                w_depth += min_edge
                if v == vert:
                    return True, w_depth
                if vert not in closed:
                    frontier.append((vert, w_depth))
            closed.append(cur_ver)
        return False

    def unifromed_search_relable(self, dfs: bool=False):
        colours = ['#ffb2b2', '#ff9999', '#ff7f7f', '#ff6666', '#ff4c4c', '#ff3232', '#ff1919', '#ff0000', '#e50000',
                   '#cc0000', '#b20000', '#990000', '#7f0000', '# 660000', '# 4c0000', '# 330000', '# 190000','# 000000']
        frontier = [self.vertices[0]]
        closed = []
        incr = 0
        re_lab = {frontier[0].label: 0}
        while len(frontier) != 0:
            if dfs: cur_ver = frontier.pop()
            else: cur_ver = frontier.pop(0)
            for ver in cur_ver.neighbours:
                if ver not in closed and ver not in frontier:
                    frontier.append(ver)
            re_lab.update({cur_ver.label: incr})
            incr += 1
            closed.append(cur_ver)
        for vertex in self.vertices:
            old = vertex.label
            new = re_lab[old]
            vertex.label = new
            if new < len(colours):
                vertex.colortext = colours[new]
            else:
                vertex.colortext = '#000000'
        return self


def single_path_graph(n):
    gr = MyGraph(False, n, True)
    for i in range(1, n):
        edge = Edge(gr.vertices[i - 1], gr.vertices[i])
        gr += edge
    return gr


def single_cycle_graph(n):
    gr = single_path_graph(n)
    new_edge = Edge(gr.vertices[n - 1], gr.vertices[0])
    gr += new_edge
    return gr


def complete_graph(size, directed):
    gr = MyGraph(directed, size, True)
    for j in range(0, size - 1):
        for k in range(j + 1, size):
            v1, v2 = gr.vertices[j], gr.vertices[k]
            gr.add_edge(Edge(v1, v2, 1))
            if directed:
                gr.add_edge(Edge(v2, v1))
    return gr

def cube_graph(dim):
    num_ver = 2**dim
    gr = MyGraph(False, num_ver, True)
    for j in range(0, dim):
        left = ((1 << dim) - 1) << (j + 1)
        mid = 1 << j
        right = mid - 1
        for i in range(0, 2**(dim - 1)):
            v1 = (i & right) + ((i << 1) & left)
            v2 = v1 + mid
            gr.add_edge(Edge(gr.vertices[v1], gr.vertices[v2], 1))
    return gr

def random_graph(n, edge_frequency):
    gr = MyGraph(False, n, True)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if random.uniform(0, 1) < edge_frequency:
                gr.add_edge(Edge(gr.vertices[i], gr.vertices[j], random.randint(1,7)))
    return gr

def ordered_degree_graph(n):
    gr = MyGraph(False, n, True)
    for i in range(0, n//2):
        for j in range(i + 1, n-i):
            gr.add_edge(Edge(gr.vertices[i], gr.vertices[j], random.randint(1,n)))
    return gr

def full_tree_graph(n):
    gr = MyGraph(False, n, True)
    i = 0
    while i < (n - 1)//2:
        gr.add_edge(Edge(gr.vertices[i], gr.vertices[i * 2 + 1]))
        gr.add_edge(Edge(gr.vertices[i], gr.vertices[i * 2 + 2]))
        i += 1
    return gr

def spanning_tree_from_seq(sequence):
    pass

def seq_from_spanning_tree(G):
    G.containscycle
    pass


if __name__ == '__main__':
    print("___________TESTING____________")
    # gr1 = single_path_graph(7)
    # gr2 = single_cycle_graph(7)
    # gr22 = single_cycle_graph(7)
    # gr3 = complete_graph(12, False)
    # G = random_graph(16, 0.24)
    # G = MyGraph(False, 10000, True)
    # gv = G.vertices
    # for j in range(0,1000):
    #     k = 10 * j
    #     G.add_edge(Edge(gv[k + 1], gv[k + 2], 1))
    #     G.add_edge(Edge(gv[k + 1], gv[k + 4], 1))
    #     G.add_edge(Edge(gv[k + 2], gv[k + 3], 1))
    #     G.add_edge(Edge(gv[k + 2], gv[k + 4], 1))
    #     G.add_edge(Edge(gv[k + 2], gv[k + 5], 1))
    #     G.add_edge(Edge(gv[k + 4], gv[k + 5], 1))
    #     G.add_edge(Edge(gv[k + 4], gv[k + 0], 1))
    #     G.add_edge(Edge(gv[k + 5], gv[k + 0], 1))
    #     G.add_edge(Edge(gv[k + 4], gv[k + 6], 1))
    #     G.add_edge(Edge(gv[k + 4], gv[k + 7], 1))
    #     G.add_edge(Edge(gv[k + 6], gv[k + 7], 1))
    #     G.add_edge(Edge(gv[k + 7], gv[k + 5], 1))
    #     G.add_edge(Edge(gv[k + 6], gv[k + 8], 1))
    #     G.add_edge(Edge(gv[k + 4], gv[k + 9], 1))
    #     if j > 0:
    #         G.add_edge(Edge(gv[k - 1], gv[k + 8], 1))
    #
    #
    #
    # F = MyGraph(False, 10000, True)
    # gv = F.vertices
    # for i in range(0,1000):
    #     k = 10 * i
    #     F.add_edge(Edge(gv[k + 1], gv[k + 2], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 3], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 4], 1))
    #     F.add_edge(Edge(gv[k + 2], gv[k + 3], 1))
    #     F.add_edge(Edge(gv[k + 2], gv[k + 5], 1))
    #     F.add_edge(Edge(gv[k + 4], gv[k + 5], 1))
    #     F.add_edge(Edge(gv[k + 5], gv[k + 0], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 5], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 6], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 7], 1))
    #     F.add_edge(Edge(gv[k + 6], gv[k + 7], 1))
    #     F.add_edge(Edge(gv[k + 7], gv[k + 2], 1))
    #     F.add_edge(Edge(gv[k + 8], gv[k + 6], 1))
    #     F.add_edge(Edge(gv[k + 1], gv[k + 9], 1))
    #     if i > 0:
    #         F.add_edge(Edge(gv[k - 1], gv[k + 8], 1))



    # G = full_tree_graph(7)
    # F = full_tree_graph(7)
    # F.unifromed_search_relable(True)

    # G = cube_graph(4)
    # G = ordered_degree_graph(16)
    # with open('./isographs/colorref_smallexample_2_49.grl', 'r') as f:
    #     G = load_graph(f)
    with open('./isographs/colorref_largeexample_6_960.grl', 'r') as f:
        L = load_graph(f, read_list=True)

    E = L[0][0]
    F = L[0][1]
    G = L[0][2]
    H = L[0][3]
    I = L[0][4]
    J = L[0][5]
    # print(G)
    # oldG = G
    # G.unifromed_search_relable()
    # print(G)
    startt = time()
    # result = G.isisomorphic(F)
    result = E.is_isomorphic_by_colour(I)
    print(result[0])
    result = F.is_isomorphic_by_colour(G)
    print(result[0])
    result = F.is_isomorphic_by_colour(H)
    print(result[0])
    result = G.is_isomorphic_by_colour(J)
    print(result[0])
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    R_graph = result[1]

    """RESULTS
    Large example 6 graphs 960 vertices:
    1 and 5 are isomorphic
    3 and 6 are isomorphic
    2 and 4 are undecided"""
    # F = oldG
    # G = random_graph(15, 0.25)
    # E = D.complement()
    # F = D + E
    # F.add_edge(Edge(F.vertices[1], F.vertices[5]))
    # with open('examplegraph2.gr', 'w') as f:
    #     save_graph(D, f)

    # with open('./samples/randomweighted.gr', 'w') as f:
    #     save_graph(G, f)

    with open('dotgraph.dot', 'w') as f:
        write_dot(R_graph, f)
    # G = nx.petersen_graph()
    # plt.subplot(121)
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()



