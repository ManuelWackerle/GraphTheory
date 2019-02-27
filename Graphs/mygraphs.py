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

    def isisomorphic(self, graph):
        length = len(self.vertices)
        if len(graph.vertices) != length:
            return False, None, None
        n_graph = self + graph
        size = len(n_graph.vertices)
        map = {}
        col_list = [-1] * size
        for v in n_graph.vertices:
            vd = v.degree
            v.colornum = vd
            v.colornew = vd
            col_list[vd] = vd
            if vd in map:
                map.get(vd).append(v)
            else:
                map.update({vd: [v]})
        map_update = self.mapcopy(map)
        old = len(map.keys())
        new = old - 1
        while old != new:
            old = new
            for k in map.keys():
                ver = map.get(k)[0]
                gcolour = self.neighbour_colours(ver)
                first = True
                for v in map[k]:
                    if self.neighbour_colours(v) != gcolour:
                        map_update[k].remove(v)
                        if first:
                            nc = self.next_col(col_list)
                            col_list[nc] = nc
                            first = False
                        v.colornew = nc
                        if nc in map_update:
                            map_update.get(nc).append(v)
                        else:
                            map_update.update({nc: [v]})
                for v in map[k]:
                    v.colornum = v.colornew
            map = self.mapcopy(map_update)
            new = len(map_update.keys())
        colours = []
        premap = {}
        mapping = {}
        for i in range(0, length):
            v = n_graph.vertices[i]
            v_col = v.colornum
            if v_col in colours:
                return "undetermined", n_graph, None
            else:
                colours.append(v_col)
                premap.update({v_col: v})
        for i in range(length, length * 2):
            u = n_graph.vertices[i]
            u_col = u.colornum
            if u_col in colours:
                colours.remove(u_col)
                mapping.update({premap.get(u_col): u})
            else:
                return False, n_graph, None
        return True, n_graph, mapping


    def colour_refinement(self):
        startt = time()
        size = len(self.vertices)
        map = {}
        colours = [-1] * size
        for v in self.vertices:
            vd = v.degree
            v.colornum = vd
            v.colornew = vd
            colours[vd] = vd
            if vd in map:
                map.get(vd).append(v)
            else:
                map.update({vd: [v]})
        map_update = self.mapcopy(map)
        old = len(map.keys())
        new = old - 1
        while old != new:
            old = new
            for k in map.keys():
                ver = map.get(k)[0]
                gcolour = self.neighbour_colours(ver)
                first = True
                for v in map[k]:
                    if self.neighbour_colours(v) != gcolour:
                        map_update[k].remove(v)
                        if first:
                            nc = self.next_col(colours)
                            colours[nc] = nc
                            first = False
                        v.colornew = nc
                        if nc in map_update:
                            map_update.get(nc).append(v)
                        else:
                            map_update.update({nc: [v]})
                for v in map[k]:
                    v.colornum = v.colornew
            map = self.mapcopy(map_update)
            new = len(map_update.keys())
        endt = time()
        print("Elapsed time in seconds:", endt - startt)
        if -1 not in colours:
            return True

    def next_col(self, colours):
        i = 0
        while colours[i] != -1:
            i += 1
        return i

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

    def make_isomorphism(self):
        new_g = MyGraph(self.directed, 0, self.simple)
        for i in range(0, len(self.vertices)):
            g_new.add_vertex(Vertex(g_new))
            iso1.update({self.vertices[i]: g_new.vertices[i]})

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
    G = MyGraph(False, 8, True)
    gv = G.vertices
    G.add_edge(Edge(gv[1], gv[2], 1))
    G.add_edge(Edge(gv[1], gv[4], 1))
    G.add_edge(Edge(gv[2], gv[3], 1))
    G.add_edge(Edge(gv[2], gv[4], 1))
    G.add_edge(Edge(gv[2], gv[5], 1))
    G.add_edge(Edge(gv[4], gv[5], 1))
    G.add_edge(Edge(gv[4], gv[0], 1))
    G.add_edge(Edge(gv[5], gv[0], 1))
    G.add_edge(Edge(gv[4], gv[6], 1))
    G.add_edge(Edge(gv[4], gv[7], 1))
    G.add_edge(Edge(gv[6], gv[7], 1))
    G.add_edge(Edge(gv[7], gv[5], 1))

    F = MyGraph(False, 8, True)
    gv = F.vertices
    F.add_edge(Edge(gv[1], gv[2], 1))
    F.add_edge(Edge(gv[1], gv[3], 1))
    F.add_edge(Edge(gv[1], gv[4], 1))
    F.add_edge(Edge(gv[2], gv[3], 1))
    F.add_edge(Edge(gv[2], gv[5], 1))
    F.add_edge(Edge(gv[4], gv[5], 1))
    F.add_edge(Edge(gv[5], gv[0], 1))
    F.add_edge(Edge(gv[1], gv[5], 1))
    F.add_edge(Edge(gv[1], gv[6], 1))
    F.add_edge(Edge(gv[1], gv[7], 1))
    F.add_edge(Edge(gv[6], gv[7], 1))
    F.add_edge(Edge(gv[7], gv[2], 1))

    # G = full_tree_graph(7)
    # F = full_tree_graph(7)
    # F.unifromed_search_relable(True)

    startt = time()
    result = G.isisomorphic(F)
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)
    print(result[0])
    H = result[1]
    # G = cube_graph(4)
    # G = ordered_degree_graph(16)
    # with open('./samples/randomweighted.gr', 'r') as f:
    #     G = load_graph(f)
    # print(G)
    # oldG = G
    # G.unifromed_search_relable()
    # print(G)

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
        write_dot(H, f)
    # G = nx.petersen_graph()
    # plt.subplot(121)
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()



