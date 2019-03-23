from vertex import Vertex, VertexPointer
from edge import Edge
from typing import IO, Tuple, List, Union, Set
from time import time
import random
from copy import deepcopy

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
            v_new = Vertex(g_new)
            v_new.original = self
            g_new.add_vertex(v_new)
            iso1.update({self.vertices[i]: g_new.vertices[i]})
        for v in other.vertices:
            v_new = Vertex(g_new)
            v_new.original = other
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

    def del_edge_unsafe(self, edge):
        edge.head.remove_incidence(edge)
        edge.tail.remove_incidence(edge)
        self._e.remove(edge)

    def del_vertex(self, vertex):
        if vertex in self._v:
            for edge in vertex.incidence:
                self.del_edge(edge)
            self._size -= 1
            self._v.remove(vertex)

    def ordered_copy(self):
        copy = MyGraph(self.directed, 0, self.simple)
        size = len(self.vertices)
        deglist = []
        for i in range(0, size):
            deglist.append([])
        for v in self.vertices:
            n_v = Vertex(copy, v.label)
            deglist[v.degree].append(n_v)
            v.p_new = n_v
        for d_list in deglist:
            for n_v in d_list:
                copy.add_vertex(n_v)
        for e in self.edges:
            new_e = Edge(e.tail.p_new, e.head.p_new)
            copy.add_edge(new_e)
        return copy

    def contains_cycle(self):
        if self.directed:
            print("The method contains_cycle is not implemented for directed graphs")
            return None
        else:
            ordered = self.ordered_copy()
            v_length = len(ordered.vertices)
            point = 0
            changed = True
            while changed:
                iter = point
                re_point = True
                changed = False
                while iter < v_length:
                    v = ordered.vertices[iter]
                    if v.degree == 1:
                        ordered.del_edge_unsafe(v.incidence[0])
                        changed = True
                    else:
                        if re_point:
                            point = iter
                            re_point = False
                    iter += 1
            if ordered.edges:
                return True
            else:
                return False

    def is_connected(self, dfs: bool=False):
        frontier = [self.vertices[0]]
        closed = []
        incr = 0
        while len(frontier) != 0:
            if dfs: cur_ver = frontier.pop()
            else: cur_ver = frontier.pop(0)
            for ver in cur_ver.neighbours:
                if ver not in closed and ver not in frontier:
                    frontier.append(ver)
            incr += 1
            closed.append(cur_ver)
        if len(closed) == len(self.vertices):
            return True
        else:
            return False

    def is_tree(self):
        return self.is_connected() and not self.contains_cycle()

    def is_forest(self):
        return not self.contains_cycle()

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


if __name__ == '__main__':
    print("___________TESTING____________")
    # gr1 = single_path_graph(7)
    # gr2 = single_cycle_graph(7)
    # gr3 = complete_graph(12, False)
    # G = random_graph(16, 0.24)
    # G = MyGraph(False, 16, True)
    # gv = G.vertices
    # G.add_edge(Edge(gv[0], gv[1], 1))
    # G.add_edge(Edge(gv[1], gv[2], 1))
    # G.add_edge(Edge(gv[2], gv[3], 1))
    # G.add_edge(Edge(gv[3], gv[0], 1))
    # G.add_edge(Edge(gv[0], gv[4], 1))
    # G.add_edge(Edge(gv[0], gv[5], 1))
    # G.add_edge(Edge(gv[0], gv[6], 1))
    # G.add_edge(Edge(gv[1], gv[7], 1))
    # G.add_edge(Edge(gv[1], gv[8], 1))
    # G.add_edge(Edge(gv[1], gv[9], 1))
    # G.add_edge(Edge(gv[2], gv[10], 1))
    # G.add_edge(Edge(gv[2], gv[11], 1))
    # G.add_edge(Edge(gv[2], gv[12], 1))
    # G.add_edge(Edge(gv[3], gv[13], 1))
    # G.add_edge(Edge(gv[3], gv[14], 1))
    # G.add_edge(Edge(gv[3], gv[15], 1))
    #
    #
    # F = MyGraph(False, 16, True)
    # gv = F.vertices
    # F.add_edge(Edge(gv[0], gv[1], 1))
    # F.add_edge(Edge(gv[1], gv[2], 1))
    # F.add_edge(Edge(gv[2], gv[3], 1))
    # F.add_edge(Edge(gv[3], gv[0], 1))
    # F.add_edge(Edge(gv[0], gv[4], 1))
    # F.add_edge(Edge(gv[0], gv[5], 1))
    # F.add_edge(Edge(gv[0], gv[6], 1))
    # F.add_edge(Edge(gv[1], gv[7], 1))
    # F.add_edge(Edge(gv[1], gv[8], 1))
    # F.add_edge(Edge(gv[1], gv[9], 1))
    # F.add_edge(Edge(gv[2], gv[10], 1))
    # F.add_edge(Edge(gv[2], gv[11], 1))
    # F.add_edge(Edge(gv[2], gv[12], 1))
    # F.add_edge(Edge(gv[3], gv[13], 1))
    # F.add_edge(Edge(gv[3], gv[14], 1))
    # F.add_edge(Edge(gv[3], gv[15], 1))

    I = full_tree_graph(31)
    J = full_tree_graph(31)
    # F.unifromed_search_relable(True)

    # E = cube_graph(3)
    # F = cube_graph(3)

    # G = ordered_degree_graph(16)
    # with open('./isographs/colorref_smallexample_2_49.grl', 'r') as f:
    #     G = load_graph(f)

    with open('./isobranchgraphs/bigtrees3.grl', 'r') as f:
        L = load_graph(f, read_list=True)
    #
    E = L[0][0]
    F = L[0][1]
    G = L[0][2]
    H = L[0][3]
    # I = L[0][4]
    # J = L[0][5]
    #
    startt = time()
    # result = E.is_isomorphic_by_colour_count(F)
    # print(result[0], result[2])
    # result = G.is_isomorphic_by_colour_count(H)
    # print(result[0], result[2])
    result = I.is_isomorphic_and_count(J, True)
    print(result[0], result[1], result[2])
    # result = F.is_isomorphic_by_colour_count(H)
    # print(result[0], result[2])

    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    startt = time()
    for i in range(0, 2000):
        result = G.count_automorphism_tree_2()
    print(result[0], result[1], result[2])
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    # startt = time()
    # result = E.is_isomorphic_by_colour(F)
    # print(result[0])
    # result = G.is_isomorphic_by_colour(H)
    # print(result[0])
    # result = E.is_isomorphic_by_colour(G)
    # print(result[0])
    # result = F.is_isomorphic_by_colour(H)
    # print(result[0])
    #
    # time_elapsed = time() - startt
    # print("Elapsed time in seconds:", time_elapsed)

    # R_graph = result[1]

    # with open('./samples/randomweighted.gr', 'w') as f:
    #     save_graph(G, f)

    # with open('dotgraph.dot', 'w') as f:
    #     write_dot(R_graph, f)




