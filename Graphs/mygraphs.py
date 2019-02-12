from Graphs.vertex import Vertex
from Graphs.edge import Edge
from typing import IO, Tuple, List, Union, Set
from Graphs.graph_io import *
import random

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
        g_new = MyGraph(self.directed, self._size, self.simple)
        iso1, iso2 = {}, {}
        for i in range(0, len(self.vertices)):
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
        frontier = [(self.vertices[0], 0)]
        closed = []
        found = False
        result = (found, -1)
        inc = 0
        re_lab = {frontier[0][0].label: 0}
        while len(frontier) != 0:
            if dfs: current = frontier.pop()
            else: current = frontier.pop(0)
            cur_ver = current[0]
            for ver in cur_ver.neighbours:
                depth = current[1] + 1
                if ver not in closed:
                    frontier.append((ver, depth))
            if cur_ver.label not in re_lab.keys():
                inc += 1
                re_lab.update({cur_ver.label: inc})
            closed.append(cur_ver)

        for vertex in self.vertices:
            old = vertex.label
            new = re_lab[old]
            vertex.label = new
            if old < len(colours):
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
            gr.add_edge(Edge(v1, v2))
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
            gr.add_edge(Edge(gr.vertices[v1], gr.vertices[v2]))
    return gr

def random_graph(n, edge_frequency):
    gr = MyGraph(False, n, True)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if random.uniform(0, 1) < edge_frequency:
                gr.add_edge(Edge(gr.vertices[i], gr.vertices[j]))
    return gr

def ordered_degree_graph(n):
    gr = MyGraph(False, n, True)
    for i in range(0, n//2):
        for j in range(i + 1, n-i):
            gr.add_edge(Edge(gr.vertices[i], gr.vertices[j]))
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
    # gr22 = single_cycle_graph(7)
    # gr3 = complete_graph(12, False)
    G = full_tree_graph(15)
    #
    # with open('./samples/negativeweightexample.gr', 'r') as f:
    #     G = load_graph(f)
    # print(G)

    # oldG = G
    G.unifromed_search_relable()
    # print(G)

    # F = oldG
    # D = random_graph(3, 0.5)
    # E = D.complement()
    # F = D + E
    # F.add_edge(Edge(F.vertices[1], F.vertices[5]))
    # with open('examplegraph2.gr', 'w') as f:
    #     save_graph(D, f)
    #
    # with open('C:/Code_PyCharm/Mod7Prac/Graphs/samples/examplegraph.gr', 'w') as f:
    #     save_graph(F, f)

    with open('mygraph.dot', 'w') as f:
        write_dot(G, f)
    #



