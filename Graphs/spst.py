import os
from Graphs.graph_io import *
import math

# Use these options to change the tests:

TEST_BELLMAN_FORD_DIRECTED = True
TEST_BELLMAN_FORD_UNDIRECTED = False
TEST_DIJKSTRA_DIRECTED = True
TEST_DIJKSTRA_UNDIRECTED = False

WRITE_DOT_FILES = True

# Use this to select the graphs to test your algorithms on:
# TestInstances = ["weightedexample.gr"]
# TestInstances=["randomplanar.gr"]
# TestInstances=["randomplanar10.gr"]
# TestInstances=["bd.gr","bbf.gr"]; WriteDOTFiles=False
# TestInstances=["bbf2000.gr"]; WriteDOTFiles=False
# TestInstances=["bbf200.gr"]
TestInstances=["negativeweightexample.gr"]
# TestInstances=["negativeweightcycleexample.gr"]
# TestInstances=["WDE100.gr","WDE200.gr","WDE400.gr","WDE800.gr","WDE2000.gr"]; WriteDOTFiles=False
# TestInstances=["weightedex500.gr"];	WriteDOTFiles=False


USE_UNSAFE_GRAPH = False

def min_i(vertex_list):
    min = vertex_list[0].dist
    index, min_i = 0, 0
    for v in vertex_list:
        if v.dist < min:
            min = v.dist
            min_i = index
        index += 1
    return min_i

def relax(edge, flip_edge: bool=False):
    no_change = False
    if flip_edge:
        head, tail = edge.tail, edge.head
    else:
        head, tail = edge.head, edge.tail
    short = tail.dist + edge.weight
    if head.dist > short:
        head.dist = short
        head.in_edge = edge
    else:
        no_change = True
    return no_change

def bellman_ford_undirected(graph, start):
    for v in graph.vertices:
        v.dist = math.inf
        v.in_edge = None
    size = len(graph.vertices)
    start.dist = 0
    for i in range(1, size):
        for e in graph.edges:
            relax(e)
            relax(e, True)

def bellman_ford_directed_i(graph, start):
    for v in graph.vertices:
        v.dist = math.inf
        v.in_edge = None
    size = len(graph.vertices)
    start.dist = 0
    for i in range(1, size):
        for e in graph.edges:
            relax(e)

def bellman_ford_directed(graph, start):
    for v in graph.vertices:
        v.dist = math.inf
        v.in_edge = None
    size = len(graph.vertices)
    start.dist = 0
    for i in range(1, size):
        halt = True
        for e in graph.edges:
            halt = relax(e) and halt
        if halt:
            return halt
    print("Negative Cycles Were Detected!")
    return halt

def dijkstra_undirected(graph, start):
    for v in graph.vertices:
        v.dist = math.inf
        v.in_edge = None
    start.dist = 0
    queue = [start]
    closed = []
    while (len(queue) != 0):
        i = min_i(queue)
        u = queue.pop(i)
        closed.append(u)
        for e in u.incidence:
            if e.tail == u:
                relax(e)
                if e.head not in queue and e.head not in closed:
                    queue.append(e.head)
            else:
                relax(e, True)
                if e.tail not in queue and e.tail not in closed:
                    queue.append(e.tail)

def dijkstra_directed(graph, start: Vertex):
    for v in graph.vertices:
        v.dist = math.inf
        v.in_edge = None
    start.dist = 0
    queue = [start]
    closed = []
    while (len(queue) != 0):
        i = min_i(queue)
        u = queue.pop(i)
        closed.append(u)
        for e in u.incidence:
            if e.tail == u:
                relax(e)
                if e.head not in queue and e.head not in closed:
                    queue.append(e.head)






##############################################################################
#
# Below is test code that does not need to be changed.
#
##############################################################################

def print_max_dist(graph):
    unreachable = False
    numreachable = 0
    sorted_vertices = sorted(graph.vertices, key=lambda v: v.label)
    remote = sorted_vertices[0]
    for v in graph.vertices:
        if v.dist == math.inf:
            unreachable = True
            # print("Vertex", v,"is unreachable")
        else:
            numreachable += 1
            if v.dist > remote.dist:
                remote = v
    print("Number of reachable vertices:", numreachable, "out of", len(graph))
    print("Largest distance:", remote.dist, "For vertex", remote)


def prepare_drawing(graph):
    for e in graph.edges:
        e.colornum = 0
    for v in graph.vertices:
        if hasattr(v, "in_edge") and v.in_edge is not None:
            v.in_edge.colornum = 1
    for v in graph:
        v.label = str(v.dist)


def do_testalg(testalg, G):
    if testalg[1]:
        print("\n\nTesting", testalg[0])
        startt = time()
        if testalg[0] == "Kruskal":
            ST = testalg[2](G)
            totalweight = 0
            for e in ST:
                totalweight += e.weight
        else:
            sorted_vertices = sorted(G.vertices, key=lambda v: v.label)
            testalg[2](G, sorted_vertices[0])
        endt = time()
        print("Elapsed time in seconds:", endt - startt)

        if testalg[0] != "Kruskal":
            print_max_dist(G)
            prepare_drawing(G)
        else:
            if len(ST) < len(G.vertices) - 1:
                print("Total weight of maximal spanning forest:", totalweight)
            else:
                print("Total weight of spanning tree:", totalweight)
            for e in G.edges:
                e.colornum = 0
            for e in ST:
                e.colornum = 1
            for v in G.vertices:
                v.label = v._label

        if WRITE_DOT_FILES:
            print(os.path.join(os.getcwd(),'outgraphs\\' + testalg[3] + '.dot'), 'w')
            with open(os.path.join(os.getcwd(),'outgraphs\\' + testalg[3] + '.dot'), 'w') as f:
                write_dot(G, f, directed=testalg[4])


if __name__ == "__main__":
    from time import time

    for FileName in TestInstances:
        # Tuple arguments below:
        # First: printable string
        # Second: Boolean value
        # Third: Function
        # Fourth: Filename
        # Fifth: Whether output should be directed
        for testalg in [("Bellman-Ford, undirected", TEST_BELLMAN_FORD_UNDIRECTED, bellman_ford_undirected,
                         "BellmanFordUndirected", False),
                        ("Bellman-Ford, directed", TEST_BELLMAN_FORD_DIRECTED, bellman_ford_directed, "BellmanFordDirected",
                         True),
                        ("Dijkstra, undirected", TEST_DIJKSTRA_UNDIRECTED, dijkstra_undirected, "DijkstraUndirected",
                         False),
                        ("Dijkstra, directed", TEST_DIJKSTRA_DIRECTED, dijkstra_directed, "DijkstraDirected", True)]:
            if USE_UNSAFE_GRAPH:
                print("\n\nLoading graph", FileName, "(Fast graph)")
                with open(os.path.join(os.getcwd(), FileName)) as f:
                    G = load_graph(f, graph.Graph)
            else:
                print("\n\nLoading graph", FileName)
                with open('./samples/' +  FileName) as f:
                    G = load_graph(f)

            for i, vertex in enumerate(list(G.vertices)):
                vertex.colornum = i
            do_testalg(testalg, G)
