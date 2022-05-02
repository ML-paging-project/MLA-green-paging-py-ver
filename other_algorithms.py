import collections
import math
import random

import networkx as nx

from lru import run_lru


def random_pick(sequence, number_of_box_kinds, miss_cost):
    print('running random picking')
    total_impact = 0
    rounds = 1000
    for _ in range(rounds):
        pointer = 0
        global_lru = collections.OrderedDict()
        while pointer < len(sequence):
            startpointer = pointer
            box = random.randint(0, number_of_box_kinds - 1)
            cache_size = 2 ** box
            box_width = miss_cost * cache_size
            # Compartmentalization
            # Load top pages from LRU stack.
            mycache = collections.OrderedDict()
            for pid in global_lru.keys():
                mycache[pid] = True
                mycache.move_to_end(pid, last=True)  ###########
                if len(mycache) == cache_size:
                    break

            pointer = run_lru(mycache, cache_size, sequence, pointer, box_width, 1, miss_cost)
            endpointer = pointer

            # Update global stack
            for x in range(startpointer, endpointer):
                if sequence[x] in global_lru.keys():
                    global_lru.move_to_end(sequence[x], last=False)
                else:
                    global_lru[sequence[x]] = True
                    global_lru.move_to_end(sequence[x], last=False)

            mi = 3 * miss_cost * cache_size * cache_size
            total_impact = total_impact + mi
    print(total_impact / rounds)
    return total_impact / rounds


def michael(sequence, number_of_box_kinds, miss_cost):
    print('running Michael\'s algorithm')
    # Michael's algorithm
    MAXBOX = number_of_box_kinds - 1
    countings = [0 for i in range(number_of_box_kinds)]
    countings[0] = 1
    currentbox = 0
    pointer = 0
    total_impact = 0
    global_lru = collections.OrderedDict()
    while pointer < len(sequence):
        startpointer = pointer
        cache_size = 2 ** currentbox
        box_width = miss_cost * cache_size
        # Compartmentalization
        # Load top pages from LRU stack.
        mycache = collections.OrderedDict()
        for pid in global_lru.keys():
            mycache[pid] = True
            mycache.move_to_end(pid, last=True)  ###########
            if len(mycache) == cache_size:
                break

        pointer = run_lru(mycache, cache_size, sequence, pointer, box_width, 1, miss_cost)
        endpointer = pointer

        # Update global stack
        for x in range(startpointer, endpointer):
            if sequence[x] in global_lru.keys():
                global_lru.move_to_end(sequence[x], last=False)
            else:
                global_lru[sequence[x]] = True
                global_lru.move_to_end(sequence[x], last=False)

        mi = 3 * miss_cost * cache_size * cache_size
        total_impact = total_impact + mi
        if currentbox == MAXBOX:
            currentbox = 0
        elif countings[currentbox] % 4 == 0:
            currentbox = currentbox + 1
        else:
            currentbox = 0
        countings[currentbox] = countings[currentbox] + 1

    # michael = [total_impact for hhh in range(num_episodes)]
    print(total_impact)
    return total_impact


############################
# run lru, mark down hits and faults
def LRU(sequence, size):
    stack = collections.OrderedDict()
    marks = []
    for r in sequence:
        if r in stack.keys():
            marks.append(True)  # a hit
        else:
            if len(stack.keys()) == size:  # memory is full
                stack.popitem(last=True)
            marks.append(False)  # a fault
            stack[r] = True
        stack.move_to_end(r, last=False)
    return marks


def opt(sequence, k, number_of_box_kinds, miss_cost):
    print('running offline opt')

    # parameters
    n = len(sequence)

    ############################
    print('building dag...')
    # find end indexes
    # initialize table
    end_index = [[0 for i in range(n)] for j in range(number_of_box_kinds)]
    # find end indexes
    for i in range(number_of_box_kinds):
        memory_size = math.floor(k / (2 ** i))
        # run lru on the whole sequence
        # with memory_size = k / (2 ** i)
        is_a_hit = LRU(sequence, memory_size)
        # find the end index for node[i][0]
        box_width = miss_cost * memory_size
        request_position = 0
        actual_running_time = 0
        while actual_running_time < box_width and request_position < n:
            if is_a_hit[request_position]:
                actual_running_time += 1
            else:
                actual_running_time += miss_cost
            request_position += 1
        end_index[i][0] = request_position - 1
        # find end indexes for other nodes in row i
        for j in range(1, n):
            if is_a_hit[j - 1]:
                actual_running_time = actual_running_time - 1
            else:
                actual_running_time = actual_running_time - miss_cost
            if actual_running_time >= box_width:
                end_index[i][j] = end_index[i][j - 1]
            else:
                request_position = end_index[i][j - 1] + 1
                while actual_running_time < box_width and request_position < n:
                    if is_a_hit[request_position]:
                        actual_running_time += 1
                    else:
                        actual_running_time += miss_cost
                    request_position += 1
                end_index[i][j] = request_position - 1
    #########################################
    # turn the table to dag
    dag = nx.DiGraph()
    for i in range(number_of_box_kinds):
        dag.add_edge('start', str(i) + '-0', weight=0)
    for i in range(number_of_box_kinds):
        for j in range(n):
            if end_index[i][j] < (n - 1):
                for ii in range(number_of_box_kinds):
                    dag.add_edge(str(i) + '-' + str(j),
                                 str(ii) + '-' + str(end_index[i][j] + 1),
                                 weight=3 * miss_cost * (k / (2 ** i)) ** 2)
            else:
                dag.add_edge(str(i) + '-' + str(j), 'end',
                             weight=3 * miss_cost * (k / (2 ** i)) ** 2)
    print('searching the shortest path...')
    nodes_sorted = list(nx.topological_sort(dag))
    build_path = {}
    for v in nodes_sorted:
        build_path[v] = (math.inf, 'null')
    build_path['start'] = (0, 'null')
    for u in nodes_sorted:
        for v in list(dag.successors(u)):
            if build_path[v][0] > build_path[u][0] + dag.edges[u, v]['weight']:
                build_path[v] = (build_path[u][0] + dag.edges[u, v]['weight'], u)
    path = ['end']
    while True:
        path.append(build_path[path[-1]][1])
        if path[-1] == 'start':
            break
    opt_impact = build_path['end'][0]
    opt_path = list(reversed(path))
    # d = nx.dijkstra_path_length(dag, source='start', target='end')
    # print(d)
    return opt_impact, opt_path
