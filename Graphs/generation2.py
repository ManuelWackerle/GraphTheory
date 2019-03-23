from time import time
from graph_io import *
from copy import deepcopy
import math


"""
Speed Ups to be implemented for generation 2:
-> implement the faster version of colour refinement: DONE
-> implement graph reduction by removal of twins:     -pending-
-> split counting function into isomorphic check + single graph automorphism count for x2 speedup
"""

def isomorphic_count(graph1: MyGraph, graph2: MyGraph = None):
    count = 0
    if graph2 == None or is_isomorphic(graph1, graph2):
        count = count_automorphisms(graph1)
    return count



def remodel_graph(graph: MyGraph):
    """
    Preprocessing of graph into
    :param graph:  graph object
    :return: "ray"       array representation of vertices. vertices represented as [col, col2, [neighbours by index]]
    :return: "partition" dictionary that maps a colour int to a set of vertices (by index in ray)
    :return: "next_col"  the value of the next available colour (integer)
    :return: "parray"    pointer array that maps an index of a vertex in ray to the actual vertex object
    """
    size = len(graph.vertices)
    ray = [None]*size       #compact array representation of a graph
    partition = {}          #mapping of colour to a set of vertices represented by index
    parray = [None]*size    #mapping of index representation of vertices to vertex objects
    deg_list = []
    for dl in range(0, size + 1):
        deg_list.append([])
    for u in graph.vertices: #order vertices in reverse order by vertex degree
        ud = u.degree
        deg_list[size - ud].append(u)
    indx, col = 0, 0
    for lo in deg_list:
        if len(lo) > 0:
            partition.update({col: set()})
            for vi in lo:
                ray[indx] = [col, col, vi]
                parray[indx] = vi
                partition[col].add(indx)
                vi.index = indx
                indx += 1
            col += 1
    next_col =  col
    for rep in ray:     #additional iteration to include neighbours in ray
        neigh = []
        for n in rep[2].neighbours_fast:
            neigh.append(n.index)
        rep[2] = neigh
    return ray, partition, next_col, parray

def remodel_graph_pair(graph1: MyGraph, graph2: MyGraph):
    size1 = len(graph1.vertices)
    if size1 != len(graph2.vertices):
        return False, "Graphs differ in size"
    size = size1 * 2
    ray = [None] * size  # compact array representation of a graph
    partition = {}  # mapping of colour to a set of vertices represented by index
    parray = [None] * size  # mapping of index representation of vertices to vertex objects
    deg_list1 = []
    deg_list2 = []
    for dl in range(0, size + 1):
        deg_list1.append([])
        deg_list2.append([])
    for u in graph1.vertices:  # order vertices in reverse order by vertex degree
        ud = u.degree
        deg_list1[size - ud].append(u)
    for u in graph2.vertices:
        ud = u.degree
        deg_list2[size - ud].append(u)
    indx1, indx2, col = 0, size1, 0
    for lo in range(0, len(deg_list1)):
        lo1 = deg_list1[lo]
        lo2 = deg_list2[lo]
        if len(lo1) > 0:
            if len(lo1) != len(lo2):
                print("Failed here")
                return False, "quickfalse: not all degrees in graphs match"
            partition.update({col: set()})
            for vi in lo1:
                ray[indx1] = [col, col, vi]
                parray[indx1] = vi
                partition[col].add(indx1)
                vi.index = indx1
                indx1 += 1
            for vi in lo2:
                ray[indx2] = [col, col, vi]
                parray[indx2] = vi
                partition[col].add(indx2)
                vi.index = indx2
                indx2 += 1
            col += 1
    next_col = col
    for rep in ray:  # additional iteration to include neighbours in ray
        neigh = []
        for n in rep[2].neighbours_fast:
            neigh.append(n.index)
        rep[2] = neigh
    return True, ray, partition, next_col, parray

def generate_queue(partition):
    max, maxk = 0, 0
    queue = []
    for key, value in partition.items():
        size = len(value)
        if size > 0:
            queue.append(key)
            if size > max:
                max = size
                maxk = key
    queue.pop(maxk)
    return queue


def refine_fast(ray, partition, next, queue):
    """
    O(n.log(m)) method for colour refinement. Second version.
    :param ray:        use construct_partition[0]
    :param partition:  use construct_partition[1]
    :param next:       use construct_partition[2]
    :return:           True if further refinement is still necessary, False otherwise
    """
    # queue = generate_queue(partition)
    while len(queue) > 0:
        iter = queue.pop(0)
        group = partition[iter]
        reform = {}
        for indx in group:
            for n in ray[indx][2]:
                vv = ray[n]
                vv[1] += 1
                if vv[0] in reform.keys():
                    reform[vv[0]].add(n)
                else:
                    reform.update({vv[0]:{n}})
        for col in reform.keys():
            remap = {}
            pset, rnset = partition[col], reform[col]
            skip, first = (len(pset) == len(rnset)), True
            for chg in reform[col]:
                vv = ray[chg]
                id = vv[1] - vv[0]
                if id in remap.keys():
                    new = remap[id]
                    if new >= 0:
                        vv[0], vv[1] = new, new
                        pset.remove(chg)
                        partition[new].add(chg)
                else:
                    if skip and first:
                        remap.update({id: -1})
                    else:
                        remap.update({id: next})
                        vv[0], vv[1] = next, next
                        partition.update({next: {chg}})
                        pset.remove(chg)
                        queue.append(next)
                        next += 1
                    first = False
    for col, pset in partition.items():
        if len(pset) > 1:
            return True, next
    return False, next

def refine_and_analyse(ray, partition, next, queue):
    while len(queue) > 0:
        iter = queue.pop(0)
        group = partition[iter]
        reform = {}
        for indx in group:
            for n in ray[indx][2]:
                vv = ray[n]
                vv[1] += 1
                if vv[0] in reform.keys():
                    reform[vv[0]].add(n)
                else:
                    reform.update({vv[0]:{n}})
        for col in reform.keys():
            remap = {}
            pset = partition[col]
            rnset = reform[col]
            skip = len(pset) == len(rnset)
            first = True
            for chg in rnset:
                vv = ray[chg]
                id = vv[1] - vv[0]
                if id in remap.keys():
                    new = remap[id]
                    if new >= 0:
                        vv[0], vv[1] = new, new
                        pset.remove(chg)
                        partition[new].add(chg)
                        if len(pset) == 1:
                            return 0, 0
                else:
                    if skip and first:
                        remap.update({id: -1})
                    else:
                        remap.update({id: next})
                        vv[0], vv[1] = next, next
                        partition.update({next: {chg}})
                        pset.remove(chg)
                        if len(pset) == 1:
                            return 0, 0
                        queue.append(next)
                        next += 1
                    first = False
    for pset in partition.values():
        if len(pset) > 2:
            return -1, next
    return 1, 0

def is_isomorphic(graph1, graph2):
    size = len(graph1.vertices)
    new_model = remodel_graph_pair(graph1, graph2) #estimated at O[6n long(n)]
    #Todo... use a class with attributes or something to avoid reassinging all these fields (ray, partition etc)
    if new_model[0]:
        ray = new_model[1]
        partition = new_model[2]
        next_col = new_model[3]
        # parray = new_model[4]
    else:
        return False, new_model[1]
    queue = generate_queue(partition)
    count, point = refine_and_analyse(ray, partition, next_col, queue)
    if count == -1:
        return recursive_search(ray, partition, point, size), "iterative search"
    else:
        return count > 0, "single recolour search"

def count_isomorphisms(graph1, graph2):
    size = len(graph1.vertices)
    new_model = remodel_graph_pair(graph1, graph2) #estimated at O[6n long(n)]
    #Todo... use a class with attributes or something to avoid reassinging all these fields (ray, partition etc)
    if new_model[0]:
        ray = new_model[1]
        partition = new_model[2]
        next_col = new_model[3]
    else:
        return False, new_model[1]
    queue = generate_queue(partition)
    count, point = refine_and_analyse(ray, partition, next_col, queue)
    if count == -1:
        iso = recursive_count(ray, partition, point, size, 0)
        return iso > 0, iso, "iterative count"
    else:
        return count > 0, count, "single recolour count"

def recursive_count(ray, partition, pointer, size, count):
    old_c = find_opt_group(partition)
    pset = partition[old_c]
    if len(pset) < 4:
        print("Something went wrong")
        return None
    else:
        v_in, matches = all_matches(pset, size)
        new_c = pointer
        for m in matches:
            saved_partition = save_partition(partition)
            ray[v_in][0], ray[v_in][1] = new_c, new_c
            ray[m][0], ray[m][1] = new_c, new_c
            pset.remove(v_in)
            pset.remove(m)
            partition.update({new_c: {v_in, m}})
            new_queue = [new_c]
            stop, point = refine_and_analyse(ray, partition, new_c + 1, new_queue)
            if stop == -1:
                count = recursive_count(ray, partition, point, size, count)
            elif stop == 1:
                count += 1
            partition = saved_partition
            pset = partition[old_c]
            reset_ray(ray, partition)
        return count

def recursive_search(ray, partition, pointer, size):
    old_c = find_opt_group(partition)
    pset = partition[old_c]
    if len(pset) < 4:
        print("Something went wrong")
        return False
    else:
        v_in, matches = all_matches(pset, size)
        new_c = pointer # + 1?
        for m in matches:
            saved_partition = save_partition(partition)
            ray[v_in][0], ray[v_in][1] = new_c, new_c
            ray[m][0], ray[m][1] = new_c, new_c
            pset.remove(v_in)
            pset.remove(m)
            partition.update({new_c: {v_in, m}})
            new_queue = [new_c]
            stop, point = refine_and_analyse(ray, partition, new_c + 1, new_queue)
            if stop == -1:
                return recursive_search(ray, partition, point, size)
            elif stop == 1:
                return True
            partition = saved_partition
            pset = partition[old_c]
            reset_ray(ray, partition)
        return False

def count_automorphisms(graph1):
    size = len(graph1.vertices)
    ray, partition, next_col, parray = remodel_graph(graph1) #estimated at O[6n long(n)]
    queue = generate_queue(partition)
    incomplete, point = refine_fast(ray, partition, next_col, queue)
    if incomplete:
        return recursive_aut_count(ray, partition, point, 0), "iterative automorphism count"
    else:
        return 1, "single automorphism"

def recursive_aut_count(ray, partition, pointer, count):
    old_c = find_aut_group(partition)
    pset = partition[old_c]
    if len(pset) < 2:
        print("Something went wrong")
        return None
    else:
        new_c = pointer
        matches = list(pset)
        for m in matches:
            saved_partition = save_partition(partition)
            ray[m][0], ray[m][1] = new_c, new_c
            pset.remove(m)
            partition.update({new_c: {m}})
            new_queue = [new_c]
            incomplete, point = refine_fast(ray, partition, new_c + 1, new_queue)
            if incomplete:
                count = recursive_aut_count(ray, partition, point, count)
            else:
                count += 1
            partition = saved_partition
            pset = partition[old_c]
            reset_ray(ray, partition)
        return count



def reset_ray(ray, partition):
    for col, pset in partition.items():
        for indx in pset:
            ray[indx][0], ray[indx][1] = col, col

def all_matches(pset, size):
    matches, pre = [], 0
    for p in pset:
        if p < size:
            pre = p
        else:
            matches.append(p)
    return pre, matches


def find_opt_group(partition):
    """
    :param partition: The coloured partition
    :return: the smallest colour group greater or equal to 4
    """
    opt = 0
    high = math.inf
    for col, pset in partition.items():
        l = len(pset)
        if l >= 4 and l <= high:
            opt = col
            high = l
    return opt

def find_aut_group(partition):
    """
    :param partition: The coloured partition
    :return: the smallest colour group greater or equal to 2
    """
    opt = 0
    high = math.inf
    for col, pset in partition.items():
        l = len(pset)
        if l >= 2 and l <= high:
            opt = col
            high = l
    return opt

def save_partition(partition):
    saved = {}
    for col, pset in partition.items():
        ns = set()
        for i in pset:
            ns.add(i)
        saved.update({col: ns})
    return saved



def re_colour_graph(partition, parray):
    for col, pset in partition.items():
        for ind in pset:
            parray[ind].colornum = col


if __name__ == '__main__':
    print("___________TESTING____________")
    from utils import *
    graph_a = full_tree_graph(40)
    graph_b = full_tree_graph(40)
    """Options: 5,10,20,40,80,160,320,640,1280,2560,5120,10240"""
    # with open('../samples/treepathgraphs/treepaths5.gr', 'r') as f:
    #     tp = load_graph(f, read_list=True)[0][0]
    # with open('../samples/treepathgraphs/treepaths5.gr', 'r') as f:
    #     tp2 = load_graph(f, read_list=True)[0][0]

    # with open('../samples/isobranchgraphs/products72.grl', 'r') as f:
    #     L = load_graph(f, read_list=True)
    # graph_f = L[0][0]
    # graph_g = L[0][1]
    # graph_h = L[0][2]
    # graph_i = L[0][3]
    # graph_j = L[0][4]
    # graph_k = L[0][5]
    # graph_l = L[0][6]
    # graph_m = L[0][7]

    with open('../samples/isobranchgraphs/products72.grl', 'r') as f:
        L = load_graph(f, read_list=True)
    graph_f = L[0][0]
    graph_g = L[0][1]
    graph_h = L[0][2]
    graph_i = L[0][3]
    graph_j = L[0][4]
    graph_k = L[0][5]
    graph_l = L[0][6]
    graph_m = L[0][7]

    startt = time()
    count = count_isomorphisms(graph_a, graph_b) #, True)
    # bool, ray, partition, next_col, parray = remodel_graph_pair(graph_f, graph_k)
    # result = refine_and_analyse(ray, partition, next_col)
    # print(result[0], result[1])
    # re_colour_graph(partition, parray)
    print(count)
    # partition = result[2]
    # for key, value in partition.items():
    #     print(key,":",value)
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    startt = time()
    count = count_automorphisms(graph_a)
    print(count)
    time_elapsed = time() - startt
    print("Elapsed time in seconds:", time_elapsed)

    with open('mydotty.dot', 'w') as f:
        write_dot(graph_b, f)


