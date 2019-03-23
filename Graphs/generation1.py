#!/bin/python3
from vertex import Vertex, VertexPointer
from time import time
from graph_io import *


def find_tree_root(partition):
    mult = 1
    found = False
    for group in partition:
        if len(group) == 1:
            root = group[0]
            found = True
        elif len(group) == 2 and group[0].is_adjacent(group[1]):
            root = group[0]
            mult = 2
            found = True
    if not found:
        root = None
    return root, mult


def collect_factors(counter, factors):
    for val in counter.values():
        factors.append(val)


def count_factors(factors):
    prod = 1
    for f in factors:
        prod *= fac(f)
    return prod


def count_factors_2(factors):
    """
    Not yet verified if this is any faster than count_factors
    """
    factmap = {1: 1, 2: 2, 3: 6, 4: 24, 5: 120, 6: 720, 7: 5040, 8: 40320, 9: 362880, 10: 3628800, 11: 39916800,
               12: 479001600}
    prod = 1
    for f in factors:
        if f in factmap.keys():
            prod *= factmap[f]
        else:
            fac = fac(f)
            prod *= fac
            factmap.update({f: fac})
    return prod


def fac(n):
    k = 0
    t = 1
    while k < n:
        k = k + 1
        t = t * k
    return t


def count_automorphism_tree(graph):
    length = len(graph.vertices)
    if not graph.is_tree() or len(graph.vertices) != length:
        print("This graph is not a tree. use a different method")
        return False, 0, "Graph is not a tree"
    part = construct_partition(graph, (length + 1) // 2)
    partition = part[0]
    result = refine_colours_single(partition, part[1])
    if not result[0]:
        factors = []
        ret = find_tree_root(partition)
        root = ret[0]
        factors.append(ret[1])
        frontier = [root]
        closed = []
        while len(frontier) != 0:
            counter = {}
            cur_ver = frontier.pop()
            for ver in cur_ver.neighbours:
                col = ver.colornum
                if col in counter.keys():
                    counter[col] += 1
                else:
                    counter.update({col: 1})
                if ver not in closed and ver not in frontier:
                    frontier.append(ver)
            collect_factors(counter, factors)
            closed.append(cur_ver)
        return True, count_factors(factors), "tree_count"
    else:
        return result[1], 0, "0 or 1"


def refine_colours_single(partition, pointer):
    point = pointer
    n_point = point + 1
    while point != n_point:
        point = n_point
        iter = 0
        while iter < n_point:
            col_group = partition[iter]
            iter += 1
            width = len(col_group)
            type = neighbour_colours(col_group[0])
            first = True
            for i in range(1, width):
                v = col_group[i]
                if neighbour_colours(v) != type:
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
                    del col_group[k]
                    partition[nc].append(y)
                    y.colornum = nc
                else:
                    k += 1
    for pair in partition:
        if len(pair) > 1:
            return False, n_point - 1
    return True, 0


def is_isomorphic_single(graph, graph2):
    size = len(graph.vertices)
    n_graph = graph + graph2
    if len(graph2.vertices) != size:
        return False, n_graph, "quickfalse1"
    partition = []
    for i in range(0, size):
        partition.append([])
    deg_list = [-1] * (size - 1)
    for u in n_graph.vertices:
        ud = u.degree
        deg_list[ud] = ud
    last = relabel_col(deg_list)
    for v in n_graph.vertices:
        vd = v.degree
        label = deg_list[vd]
        v.colornum = label
        v.colornew = label
        partition[label].append(v)
    point = last
    n_point = point + 1
    while point != n_point:
        point = n_point
        iter = 0
        while iter < n_point:
            col_group = partition[iter]
            iter += 1
            width = len(col_group)
            if width > 1:
                type = neighbour_colours(col_group[0])
                first = True
                for i in range(1, width):
                    v = col_group[i]
                    if neighbour_colours(v) != type:
                        if first:
                            nc = n_point
                            n_point += 1
                            # if n_point > size:
                            #     return False, n_graph, "quickfalse3"
                            first = False
                        v.colornew = nc
                k = 1
                for j in range(1, width):
                    y = col_group[k]
                    nc = y.colornew
                    if y.colornum != nc:
                        del col_group[k]
                        partition[nc].append(y)
                        y.colornum = nc
                    else:
                        k += 1
            else:
                return False, n_graph, "quickfalse2"
    for pair in partition:
        l = len(pair)
        if l == 1:
            return False, n_graph
        if l > 2:
            return "undetermined", n_graph
    return True, n_graph, "group multiple"


def construct_partition(n_graph, size):
    partition = []
    for i in range(0, size):
        partition.append([])
    deg_list = [-1] * size
    for u in n_graph.vertices:
        ud = u.degree
        deg_list[ud] = ud
    last = relabel_col(deg_list)
    for v in n_graph.vertices:
        vd = v.degree
        label = deg_list[vd]
        vp = VertexPointer(v)
        v.pointedby = vp
        v.original = v.original
        v.colornum = label
        v.colornew = label
        partition[label].append(v)
    return partition, last


def is_isomorphic_and_count(graph, graph2, dont_count: bool = False):
    length = len(graph.vertices)
    n_graph = graph + graph2
    if len(graph2.vertices) != length:
        return False, 0, "immediate"
    size = length * 1
    part = construct_partition(n_graph, size)
    partition = part[0]
    result = refine_colours(partition, part[1])
    if not result[0]:
        iso = recursive_count(partition, result[1], 0, dont_count)
        if dont_count:
            return iso > 0, iso, "True/False only"
        else:
            return iso > 0, iso, "iterative count"
    else:
        return result[1], 0, "single recolour count"


def recursive_count(partition, pointer, count, terminate):
    old_c = find_opt_group(partition)
    group = partition[old_c]
    if len(group) > 2:
        saved = save_partition(partition)
        matches = all_matches(group)
        v_in = group[0]
        new_c = pointer + 1
        i = 0
        for m in matches:
            i += 1
            match = group[m]
            old_c = v_in.colornum
            v_in.colornum, v_in.colornew = new_c, new_c
            match.colornum, match.colornew = new_c, new_c
            del partition[old_c][0]
            del partition[old_c][m - 1]
            partition[new_c].append(v_in)
            partition[new_c].append(match)
            result = refine_colours(partition, new_c)
            undetermined = not result[0]
            iso_or_point = result[1]
            if undetermined:
                count = recursive_count(partition, iso_or_point, count, terminate)
                if terminate:
                    return count
            else:
                if iso_or_point:
                    count += 1
                    if terminate:
                        return count
            partition = restore_partition(saved)
            group = partition[old_c]
        return count
    else:
        return count


def refine_colours(partition, pointer):
    point = pointer
    n_point = point + 1
    while point != n_point:
        point = n_point
        iter = 0
        while iter < n_point:
            col_group = partition[iter]
            iter += 1
            width = len(col_group)
            if width > 1:
                type = neighbour_colours(col_group[0])
                first = True
                for i in range(1, width):
                    v = col_group[i]
                    if neighbour_colours(v) != type:
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
                        del col_group[k]
                        partition[nc].append(y)
                        y.colornum = nc
                    else:
                        k += 1
            else:
                return True, False
    for pair in partition:
        if len(pair) > 2:
            return False, n_point - 1
    return True, True


def all_matches(group):
    matches = []
    v_in = group[0]
    orig = v_in.original
    for pair in range(0, len(group)):
        v_out = group[pair]
        if v_out.original != orig:
            matches.append(pair)
    return matches


def find_opt_group(partition):
    opt = 0
    lng = len(partition)
    high = lng + 1
    for g in range(0, lng):
        group = partition[g]
        l = len(group)
        if l >= 4 and l <= high:
            opt = g
            high = len(group)
    return opt


def save_partition(partition):
    pointer_partition = []
    for group in partition:
        n_group = []
        for v in group:
            vp = VertexPointer(v)
            n_group.append(vp)
        pointer_partition.append(n_group)
    return pointer_partition


def restore_partition(saved_partition):
    restored = []
    for c_num in range(0, len(saved_partition)):
        p_group = saved_partition[c_num]
        n_group = []
        for l in range(0, len(p_group)):
            vp = p_group[l]
            vp.vertex.colornum = c_num
            vp.vertex.colornew = c_num
            n_group.append(vp.vertex)
        restored.append(n_group)
    return restored


def relabel_col(col_list):
    n_label = 0
    for i in range(0, len(col_list)):
        if col_list[i] != -1:
            col_list[i] = n_label
            n_label += 1
    return n_label - 1


def neighbour_colours(vertex: Vertex):
    group = vertex.neighbours
    gcolour = []
    for g in group:
        gcolour.append(g.colornum)
    s1 = sorted(gcolour)
    return s1

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
    graph_i = full_tree_graph(31)
    graph_j = full_tree_graph(31)
    # with open('./isobranchgraphs/products72.grl', 'r') as f:
    #     L = load_graph(f, read_list=True)
    # graph_i = L[0][0]
    # graph_j = L[0][1]
    # graph_k = L[0][2]
    # graph_p = L[0][3]

    with open('./isobranchgraphs/bigtrees3.grl', 'r') as f:
        L = load_graph(f, read_list=True)
    #
    graph_e = L[0][0]
    graph_f = L[0][1]
    graph_g = L[0][2]
    graph_h = L[0][3]
    # graph_i = L[0][4]
    # graph_j = L[0][5]
    #
    startt = time()
    result = is_isomorphic_and_count(graph_i, graph_j)
    print(result[0], result[1], result[2])
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    startt = time()
    result = count_automorphism_tree(graph_g)
    print(result[0], result[1], result[2])
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)




