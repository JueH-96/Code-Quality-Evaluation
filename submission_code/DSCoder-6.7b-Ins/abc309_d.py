import sys
from collections import defaultdict
from heapq import heappop, heappush

def read_input():
    N1, N2, M = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]
    return N1, N2, M, edges

def solve(N1, N2, M, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    dist = [float('inf')] * (N1 + N2 + 1)
    dist[1] = 0
    heap = [(0, 1)]

    while heap:
        d, node = heappop(heap)
        if dist[node] < d:
            continue
        for neighbor in graph[node]:
            if neighbor <= N1:
                new_dist = d + 1
            else:
                new_dist = d
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heappush(heap, (new_dist, neighbor))

    return dist[N1 + N2]

N1, N2, M, edges = read_input()
print(solve(N1, N2, M, edges))