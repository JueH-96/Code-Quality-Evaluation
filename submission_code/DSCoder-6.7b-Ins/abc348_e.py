import sys
from collections import defaultdict
from heapq import heappop, heappush

def read_input():
    N = int(sys.stdin.readline().strip())
    graph = defaultdict(list)
    for _ in range(N - 1):
        a, b = map(int, sys.stdin.readline().strip().split())
        graph[a].append(b)
        graph[b].append(a)
    C = list(map(int, sys.stdin.readline().strip().split()))
    return N, graph, C

def dijkstra(graph, start):
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        dist, node = heappop(heap)
        if distances[node] < dist:
            continue
        for neighbor in graph[node]:
            new_dist = dist + 1
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heappush(heap, (new_dist, neighbor))
    return distances

def solve():
    N, graph, C = read_input()
    distances = dijkstra(graph, 1)
    min_f = float('inf')
    for x in range(1, N + 1):
        f = sum(C[i - 1] * distances[i] for i in range(1, N + 1))
        min_f = min(min_f, f)
    print(min_f)

solve()