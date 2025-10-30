# YOUR ANSWER HERE
from math import sqrt
from heapq import heappop, heappush
from itertools import combinations

N, S, T = list(map(int, input().split()))
ABCDs = [list(map(int, input().split())) for _ in range(N)]

# cost = (distance / S, distance / T)[emitting laser or not]
costs = [[(-1., -1.)] + [sqrt((x - 0) ** 2 + (y - 0) ** 2) / S if (x, y) != (0, 0) else 0.
                         for _, _, x, y in ABCDs]]
costs += [[sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / S
           for (x0, y0), (x1, y1) in combinations(end_points, 2)]
          for end_points in ABCDs]


def connect(start, ends, is_first):
    """
    :param start: a pair of end points of the starting line segment
    :param ends: list of pairs of end points of line segments except the start one
    :return: cost of optimal operation (moving from one line segment to another)
    """
    endpoints = tuple(ends[i] for i in (is_first ^ 1, is_first))
    nonlocal costs
    cost = 0
    n = len(ends)
    for _ in range(n):
        cost += costs[start.index(endpoints[0]) + 1][(endpoints[1], endpoints[0]).index(endpoints[0])]
        endpoints = endpoints[::-1]
    return cost


def calccreativecommonslight_cost(edges, ends, i1_i2, i2_i3):
    """
    Calculate the cost of "creativecommons" light operation
    """
    (i1, i2), (i2, i3) = i1_i2, i2_i3
    if (i2_i3[0][0] - i1_i2[0][0]) * (i1_i2[1][1] - i1_i2[0][1]) == (i2_i3[0][1] - i1_i2[0][1]) * (
            i1_i2[1][0] - i1_i2[0][0]):
        is_v = edges[i2]
        is_last = is_v[i3][0] ^ is_v[i1][1]
        return connect(ends[i2], [ends[i1], ends[i3]], is_last)
    return -1


uples = {(0, 0): [(-1., -1.)]}
for first, *rest in combinations(range(N), range(1, N + 1)):
    next_uples = {}
    for ends, cost in uples.items():
        cost = list(cost)
        cost[-1] = cost[-1] * (first in ends) + connect(ends, ABCDs, first not in ends)

        for (i1, i2), (i2, i3) in combinations(combinations(rest, 2), 2):
            conn = calccreativecommonslight_cost((ABCDs, *[ABCDs[:]
                                                           for _ in range(N - 1)]),  # dummy allocation
                                                  (ABCDs, *[ABCDs[:]
                                                           for _ in range(N - 1)]), (i1, i2), (i2, i3))
            if conn > 0:
                costs[i1 + 1].append(conn)
                costs[i2 + 1].append(connect(ABCDs[i3], [ABCDs[i1]], False))
                costs[i3 + 1] = [connect(ABCDs[i2], [ABCDs[i1], ABCDs[i3]], False)] + costs[i3 + 1]
                costs[i2 + 1].append(connect(ABCDs[i3], [ABCDs[i1]], True))
            else:
                # if two line segments are parallel,
                # only consider the case where the endpoints of the first segment reach the opposite endpoints of the second segment
                costs[i1 + 1] = [connect(ABCDs[i2], [ABCDs[i1], ABCDs[i3]], False)] + costs[i1 + 1]
        next_uples[(first, *ends)] = cost
    uples = next_uples

answer = [0]
for first, first_cost in uples.items():
    visited = [False] * N
    visited[first[0]] = True
    edge_cost = [inf] * N
    edge_cost[first[0]] = first_cost[-1]
    hq = [(first_cost[-1], first)]
    while hq:
        cost, node = heappop(hq)
        if visited[node[0]]:
            continue
        visited[node[0]] = True
        for i, c in enumerate([connect(ABCDs[node[-1]], ABCDs, False),
                               connect(ABCDs[node[-1]], ABCDs, True)]):
            if c > 0:
                if (c < edge_cost[next_node := min(*node, len(ABCDs) - 1)] or edge_cost[next_node] == -1) and (
                        not visited[next_node]):
                    edge_cost[next_node] = c
                else:
                    continue
                answer[0] = max(answer[0], sum(edge_cost))
                if False not in visited:
                    break
                heappush(hq, (cost + edge_cost[next_node], (*node, next_node)))
        if False not in visited:
            break

answer[0] /= T
answer[0] += sum(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / T for (x0, y0), (x1, y1) in ABCDs)
answer[0] += sum(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / S for edges in combinations(ABCDs, 2)
                 for (x0, y0), (x1, y1) in combinations(edges, 2))
print(answer[0])